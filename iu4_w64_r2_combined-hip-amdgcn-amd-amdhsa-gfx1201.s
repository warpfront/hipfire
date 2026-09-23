	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	quantize_int4_mmq_ds128 ; -- Begin function quantize_int4_mmq_ds128
	.globl	quantize_int4_mmq_ds128
	.p2align	8
	.type	quantize_int4_mmq_ds128,@function
quantize_int4_mmq_ds128:                ; @quantize_int4_mmq_ds128
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x24
	s_load_b64 s[10:11], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_lt_i32 ttmp7, s11
	v_mad_co_u64_u32 v[5:6], null, ttmp9, s2, v[0:1]
	s_cselect_b64 s[2:3], -1, 0
	v_lshlrev_b32_e32 v6, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], s[2:3], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[2:3]
	s_cbranch_execz .LBB0_16
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_or_b32_e32 v1, 3, v6
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s8, ttmp7
	v_cmp_gt_i32_e32 vcc, s10, v1
	v_mov_b32_e32 v1, 0
	s_ashr_i32 s9, ttmp7, 31
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_ashrrev_i32_e32 v7, 31, v6
	s_ashr_i32 s3, s10, 31
	s_mov_b32 s2, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[2:3], s[8:9]
	v_lshlrev_b64_e32 v[1:2], 2, v[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 2
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc, s2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s3, v2, vcc
	global_load_b128 v[1:4], v[1:2], off
.LBB0_3:                                ; %._crit_edge
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_wait_loadcnt 0x0
	v_max_num_f32_e64 v10, |v4|, |v4|
	s_mov_b32 s9, 0
	s_mov_b64 s[12:13], exec
	v_mbcnt_hi_u32_b32 v6, -1, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_and_b32_e32 v7, 0x60, v6
	v_xor_b32_e32 v8, 16, v6
	v_xor_b32_e32 v11, 8, v6
	v_add_nc_u32_e32 v9, 32, v7
	v_max3_num_f32 v7, |v1|, |v2|, |v3|
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_lt_u32_e32 vcc, v8, v9
	v_max_num_f32_e32 v10, v7, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v8, v6, v8, vcc
	v_cmp_lt_u32_e32 vcc, v11, v9
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v7, 2, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v6, v11, vcc
	ds_bpermute_b32 v8, v7, v10
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v8, v8
	v_lshlrev_b32_e32 v8, 2, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_max_num_f32_e32 v11, v10, v12
	v_xor_b32_e32 v12, 4, v6
	ds_bpermute_b32 v10, v8, v11
	v_cmp_lt_u32_e32 vcc, v12, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v6, v12, vcc
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v13, v10, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v10, 2, v12
	v_max_num_f32_e32 v12, v11, v13
	v_xor_b32_e32 v13, 2, v6
	ds_bpermute_b32 v11, v10, v12
	v_cmp_lt_u32_e32 vcc, v13, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v6, v13, vcc
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v14, v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v11, 2, v13
	v_max_num_f32_e32 v12, v12, v14
	v_xor_b32_e32 v14, 1, v6
	ds_bpermute_b32 v13, v11, v12
	v_cmp_lt_u32_e32 vcc, v14, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v6, v14, vcc
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v9, 2, v6
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v13, v13, v13
	v_max_num_f32_e32 v6, v12, v13
	ds_bpermute_b32 v12, v9, v6
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_max_num_f32_e32 v12, v6, v12
	v_mov_b32_e32 v6, 1.0
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_6
; %bb.4:                                ; %.preheader76.i
	v_div_scale_f32 v6, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc, v12, 0x40e00000, v12
	s_mov_b32 s10, 0x40e00000
	s_mov_b32 s14, 0xc1000000
	v_rcp_f32_e32 v13, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v6, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v13
	v_fma_f32 v16, -v6, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v13
	v_fma_f32 v6, -v6, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v6, v6, v13, v14
	v_mov_b32_e32 v14, 0x7149f2ca
	v_div_fixup_f32 v13, v6, 0x40e00000, v12
	v_mov_b32_e32 v6, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v13, 0.5, v13
.LBB0_5:                                ; %.preheader.preheader.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s0, s9
	s_add_co_i32 s9, s9, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	s_cmp_lg_u32 s9, 8
	s_wait_alu depctr_sa_sdst(0)
	v_div_scale_f32 v15, null, s10, s10, s0
	v_div_scale_f32 v16, vcc, s0, 0x40e00000, s0
	v_rcp_f32_e32 v17, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v15, v17, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v19, v15, v18, v16
	v_fmac_f32_e32 v18, v19, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v15, v18
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s0
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v13, v15
	v_div_scale_f32 v16, null, v15, v15, v1
	v_div_scale_f32 v18, null, v15, v15, v2
	v_div_scale_f32 v20, null, v15, v15, v3
	v_div_scale_f32 v22, null, v15, v15, v4
	v_div_scale_f32 v17, vcc, v1, v15, v1
	v_rcp_f32_e32 v24, v16
	v_rcp_f32_e32 v25, v18
	v_rcp_f32_e32 v26, v20
	v_rcp_f32_e32 v27, v22
	v_div_scale_f32 v19, s[0:1], v2, v15, v2
	v_div_scale_f32 v21, s[2:3], v3, v15, v3
	s_wait_kmcnt 0x0
	v_div_scale_f32 v23, s[4:5], v4, v15, v4
	v_fma_f32 v28, -v16, v24, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v29, -v18, v25, 1.0
	v_fma_f32 v30, -v20, v26, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v31, -v22, v27, 1.0
	v_fmac_f32_e32 v24, v28, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v25, v29, v25
	v_fmac_f32_e32 v26, v30, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v31, v27
	v_mul_f32_e32 v28, v17, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v29, v19, v25
	v_mul_f32_e32 v30, v21, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v31, v23, v27
	v_fma_f32 v32, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v33, -v18, v29, v19
	v_fma_f32 v34, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v35, -v22, v31, v23
	v_fmac_f32_e32 v28, v32, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v29, v33, v25
	v_fmac_f32_e32 v30, v34, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v31, v35, v27
	v_fma_f32 v16, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v17, -v18, v29, v19
	v_fma_f32 v18, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v19, -v22, v31, v23
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v24, v28
	s_mov_b64 vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v25, v29
	s_mov_b64 vcc, s[2:3]
	v_div_fixup_f32 v16, v16, v15, v1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v26, v30
	s_mov_b64 vcc, s[4:5]
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v19, v27, v31
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v3
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v19, v19, v15, v4
	v_med3_num_f32 v16, v16, s14, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s14, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v19, v19
	v_fma_f32 v16, -v16, v15, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s14, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v19, v19, s14, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v3
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v19, v15, v4
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_cndmask_b32_e32 v6, v6, v15, vcc
	s_cbranch_scc1 .LBB0_5
.LBB0_6:                                ; %Flow39
	s_or_b64 exec, exec, s[12:13]
	v_cmp_neq_f32_e64 s[0:1], 0, v12
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_div_scale_f32 v13, null, v6, v6, v1
	s_wait_kmcnt 0x0
	s_mov_b32 s4, 0xc1000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v14, v13
	v_fma_f32 v15, -v13, v14, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc, v1, v6, v1
	v_mul_f32_e32 v16, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, -v13, v16, v15
	v_fmac_f32_e32 v16, v17, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v13, v13, v14, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v13, v6, v1
	v_rndne_f32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v1, v1, s4, 0x40e00000
	v_cvt_i32_f32_e32 v13, v1
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_div_scale_f32 v1, null, v6, v6, v2
	s_wait_kmcnt 0x0
	s_mov_b32 s4, 0xc1000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v1
	v_fma_f32 v14, -v1, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc, v2, v6, v2
	v_mul_f32_e32 v15, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v16, -v1, v15, v14
	v_fmac_f32_e32 v15, v16, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v1, v1, v12, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v1, v6, v2
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v1, v1, s4, 0x40e00000
	v_cvt_i32_f32_e32 v12, v1
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v1, 0
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_div_scale_f32 v1, null, v6, v6, v3
	s_wait_kmcnt 0x0
	s_mov_b32 s4, 0xc1000000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v14, v1
	v_fma_f32 v15, -v1, v14, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc, v3, v6, v3
	v_mul_f32_e32 v16, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, -v1, v16, v15
	v_fmac_f32_e32 v16, v17, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v1, -v1, v16, v15
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v1, v1, v14, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v1, v1, v6, v3
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maxmin_num_f32 v1, v1, s4, 0x40e00000
	v_cvt_i32_f32_e32 v1, v1
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_div_scale_f32 v2, null, v6, v6, v4
	s_mov_b32 s0, 0xc1000000
	v_rcp_f32_e32 v3, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v2, v3, 1.0
	v_fmac_f32_e32 v3, v14, v3
	v_div_scale_f32 v14, vcc, v4, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v3
	v_fma_f32 v16, -v2, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v3
	v_fma_f32 v2, -v2, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v2, v2, v3, v15
	v_div_fixup_f32 v2, v2, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s0, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v2
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	v_add_nc_u32_e32 v3, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add3_u32 v3, v3, v1, v2
	ds_bpermute_b32 v4, v7, v3
	v_ashrrev_i32_e32 v7, 31, v5
	v_lshrrev_b32_e32 v7, 27, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v5, v5, v7
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v8, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v10, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v3, v4
	v_ashrrev_i32_e32 v3, 5, v5
	ds_bpermute_b32 v5, v11, v7
	v_mad_co_i64_i32 v[3:4], null, v3, s11, 0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[14:15], null, 0x48, v3, s[6:7]
	v_mad_co_u64_u32 v[15:16], null, 0x48, v4, v[15:16]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v7, v5
	v_and_b32_e32 v5, 15, v1
	v_and_b32_e32 v7, 15, v13
	ds_bpermute_b32 v4, v9, v3
	v_and_b32_e32 v9, 31, v0
	v_mad_co_i64_i32 v[0:1], null, 0x48, s8, v[14:15]
	v_lshl_or_b32 v2, v2, 4, v5
	v_lshl_or_b32 v5, v12, 4, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v7, 1, v9
	v_lshlrev_b16 v2.l, 8, v2.l
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b16 v2.h, 0xff, v5.l
	v_add_co_u32 v7, vcc, v0, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v1, vcc
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b16 v2.l, v2.h, v2.l
	v_cmp_eq_u32_e32 vcc, 0, v9
	global_store_b16 v[7:8], v2, off offset:8
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v3, v4
	global_store_b64 v[0:1], v[6:7], off
.LBB0_16:                               ; %_Z26quantize_block_i4_128_wavePKfP12block_i4_128i.exit
	s_endpgm
.Lfunc_end0:
	.size	quantize_int4_mmq_ds128, .Lfunc_end0-quantize_int4_mmq_ds128
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel quantize_int4_mmq_ds128
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 15
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-quantize_int4_mmq_ds128)<<4)&4080)>>4
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
	.set .Lquantize_int4_mmq_ds128.num_vgpr, 36
	.set .Lquantize_int4_mmq_ds128.num_agpr, 0
	.set .Lquantize_int4_mmq_ds128.numbered_sgpr, 15
	.set .Lquantize_int4_mmq_ds128.num_named_barrier, 0
	.set .Lquantize_int4_mmq_ds128.private_seg_size, 0
	.set .Lquantize_int4_mmq_ds128.uses_vcc, 1
	.set .Lquantize_int4_mmq_ds128.uses_flat_scratch, 0
	.set .Lquantize_int4_mmq_ds128.has_dyn_sized_stack, 0
	.set .Lquantize_int4_mmq_ds128.has_recursion, 0
	.set .Lquantize_int4_mmq_ds128.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2348
; TotalNumSgprs: 17
; NumVgprs: 36
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 36
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
	.protected	iu4_w64_r2              ; -- Begin function iu4_w64_r2
	.globl	iu4_w64_r2
	.p2align	8
	.type	iu4_w64_r2,@function
iu4_w64_r2:                             ; @iu4_w64_r2
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[40:43], s[0:1], 0x18
	s_lshl_b32 s2, ttmp9, 7
	s_lshl_b32 s14, ttmp7, 7
	s_clause 0x1
	s_load_b128 s[48:51], s[0:1], 0x0
	s_load_b64 s[38:39], s[0:1], 0x10
	v_lshrrev_b32_e32 v5, 2, v0
	v_lshlrev_b32_e32 v6, 7, v0
	v_and_b32_e32 v7, 60, v0
	v_and_b32_e32 v1, 3, v0
	v_bfe_u32 v2, v0, 1, 1
	v_lshrrev_b32_e32 v8, 5, v0
	v_or_b32_e32 v10, s2, v5
	v_and_or_b32 v6, 0x80, v6, v7
	v_or_b32_e32 v7, s14, v5
	v_lshlrev_b32_e32 v1, 3, v1
	v_lshrrev_b32_e32 v3, 1, v0
	v_lshrrev_b32_e32 v4, 6, v0
	v_and_b32_e32 v96, 15, v0
	v_lshlrev_b32_e32 v95, 2, v0
	v_cmp_gt_u32_e64 s[24:25], 0x100, v0
	v_and_b32_e32 v9, 0x60, v3
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s2, s40
	v_cmp_gt_i32_e64 s[16:17], s42, v7
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_ge_i32 s14, s42
	v_or_b32_e32 v34, s2, v96
	s_cselect_b64 s[4:5], -1, 0
	s_ashr_i32 s3, s41, 31
	s_add_co_i32 s26, s42, -1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s3, s3, 24
	s_add_co_i32 s6, s40, -1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s41, s3
	s_or_b64 s[54:55], s[0:1], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s33, s3, 8
	s_cmp_gt_i32 s41, 0xff
	v_min_i32_e32 v11, s26, v7
	s_cselect_b64 s[52:53], -1, 0
	s_ashr_i32 s15, s14, 31
	s_ashr_i32 s41, s40, 31
	s_ashr_i32 s3, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[40:41], s[14:15]
	s_lshl_b64 s[4:5], s[2:3], 2
	s_lshl_b64 s[0:1], s[0:1], 2
	s_ashr_i32 s27, s42, 31
	s_add_nc_u64 s[0:1], s[38:39], s[0:1]
	v_min_i32_e32 v12, s6, v10
	s_add_nc_u64 s[44:45], s[0:1], s[4:5]
	s_add_co_i32 s0, s2, 0x80
	s_and_b32 s45, s45, 0xffff
	s_cmp_gt_i32 s0, s40
	v_mad_co_u64_u32 v[36:37], null, 0x48, v11, v[1:2]
	v_and_or_b32 v11, v8, 6, v2
	v_or_b32_e32 v7, 64, v7
	v_or_b32_e32 v8, 8, v8
	s_cselect_b64 s[0:1], -1, 0
	s_add_co_i32 s3, s14, 0x80
	v_or_b32_e32 v10, 64, v10
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s3, s42
	s_mul_i32 s3, s33, 0x88
	v_lshl_or_b32 v99, v11, 8, v6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[37:38], null, s3, v12, v[1:2]
	v_min_i32_e32 v11, s26, v7
	v_and_or_b32 v2, v8, 14, v2
	v_min_i32_e32 v10, s6, v10
	v_cmp_gt_i32_e64 s[18:19], s42, v7
	v_or_b32_e32 v7, s14, v3
	v_and_b32_e32 v8, 1, v0
	v_mad_co_u64_u32 v[38:39], null, 0x48, v11, v[1:2]
	v_mad_co_u64_u32 v[39:40], null, s3, v10, v[1:2]
	v_lshl_or_b32 v100, v2, 8, v6
	v_mad_co_i64_i32 v[1:2], null, 0x48, v7, s[50:51]
	v_lshlrev_b32_e32 v90, 2, v8
	v_or_b32_e32 v6, s2, v3
	v_lshlrev_b32_e32 v3, 3, v3
	v_lshrrev_b32_e32 v0, 4, v0
	s_cselect_b64 s[4:5], -1, 0
	v_and_b32_e32 v98, 12, v5
	v_min_i32_e32 v10, s6, v6
	v_add_co_u32 v52, vcc, v1, v90
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v53, null, 0, v2, vcc
	v_add3_u32 v91, 0, v3, v90
	v_mul_u32_u24_e32 v2, 0x500, v4
	v_lshlrev_b32_e32 v3, 2, v96
	v_cmp_gt_i32_e64 s[22:23], s40, v6
	v_mul_u32_u24_e32 v4, 0x50, v96
	v_lshlrev_b32_e32 v6, 2, v0
	v_mad_co_u64_u32 v[32:33], null, s3, v10, s[48:49]
	v_add3_u32 v97, 0, v2, v3
	v_or_b32_e32 v2, 32, v34
	v_ashrrev_i32_e32 v1, 31, v10
	v_add3_u32 v86, 0, v4, v6
	v_or_b32_e32 v4, 0x60, v34
	v_or_b32_e32 v3, 64, v34
	v_cmp_gt_i32_e64 s[12:13], s40, v2
	v_or_b32_e32 v2, 16, v34
	s_or_b64 s[36:37], s[0:1], s[4:5]
	v_add_co_u32 v42, s[0:1], s50, v36
	v_cmp_gt_i32_e64 s[8:9], s40, v4
	v_mul_lo_u32 v4, s40, v0
	v_add_co_ci_u32_e64 v43, null, s51, 0, s[0:1]
	v_add_co_u32 v46, s[0:1], s48, v37
	v_and_b32_e32 v88, 0xfc, v95
	v_or_b32_e32 v5, v98, v9
	v_cmp_gt_i32_e64 s[10:11], s40, v3
	v_or_b32_e32 v3, 48, v34
	v_cmp_gt_i32_e64 s[6:7], s40, v2
	v_or_b32_e32 v2, 0x50, v34
	v_mad_co_u64_u32 v[40:41], null, s3, v1, v[33:34]
	v_or_b32_e32 v1, 0x70, v34
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v47, null, s49, 0, s[0:1]
	v_add_co_u32 v44, s[0:1], s50, v38
	v_mad_co_i64_i32 v[50:51], null, s3, v10, s[48:49]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, s51, 0, s[0:1]
	v_add_co_u32 v48, s[0:1], s48, v39
	v_lshl_or_b32 v89, v9, 5, v88
	v_cmp_gt_i32_e64 s[20:21], s42, v7
	v_lshlrev_b32_e32 v92, 4, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v49, null, s49, 0, s[0:1]
	v_min_i32_e32 v93, s26, v7
	v_cmp_gt_i32_e64 s[4:5], s40, v3
	v_lshl_add_u32 v94, v5, 3, 0
	v_cmp_gt_i32_e64 s[2:3], s40, v2
	v_add_lshl_u32 v33, v4, v96, 2
	v_or_b32_e32 v87, s14, v0
	v_cmp_gt_i32_e64 s[14:15], s40, v34
	v_ashrrev_i32_e32 v35, 31, v34
	v_cmp_gt_i32_e64 s[0:1], s40, v1
	s_mov_b32 s46, -1
	s_cmp_eq_u32 s43, 0
	s_mov_b32 s43, s27
	s_mov_b32 s47, 0x31004000
	s_cbranch_scc1 .LBB1_17
; %bb.1:
	s_and_b64 vcc, exec, s[54:55]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_3
; %bb.2:                                ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb1ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.1.critedge.i
	s_mov_b64 s[26:27], 0
.LBB1_3:                                ; %Flow2281
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_177
; %bb.4:                                ; %.preheader514.i.i
	s_and_saveexec_b64 s[26:27], s[24:25]
	s_cbranch_execz .LBB1_8
; %bb.5:                                ; %.lr.ph.i.i
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[28:29], s[20:21]
	s_cbranch_execz .LBB1_7
; %bb.6:
	global_load_b32 v0, v[52:53], off
.LBB1_7:                                ; %.preheader511.loopexit.i.i
	s_or_b64 exec, exec, s[28:29]
	global_load_b32 v1, v[50:51], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v92, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[22:23]
	ds_store_2addr_stride64_b32 v91, v0, v1 offset0:48 offset1:56
.LBB1_8:                                ; %Flow2280
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	global_load_b64 v[0:1], v[42:43], off offset:8
	global_load_b64 v[2:3], v[46:47], off offset:8
	global_load_b64 v[4:5], v[44:45], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_add_nc_u32_e32 v101, 0, v99
	v_add_nc_u32_e32 v102, 0, v100
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s27, 0
	s_and_not1_b64 vcc, exec, s[52:53]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[16:17]
	v_cndmask_b32_e64 v1, 0, v1, s[16:17]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[18:19]
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	ds_store_b32 v101, v2 offset:4096
	ds_store_2addr_b32 v101, v0, v1 offset1:16
	ds_store_b32 v101, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v102, v6 offset:4096
	ds_store_2addr_b32 v102, v4, v5 offset1:16
	ds_store_b32 v102, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_19
; %bb.9:                                ; %.preheader510.lr.ph.i.i
	v_lshl_add_u32 v134, v96, 3, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	s_movk_i32 s60, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s62, 0x3800
	s_mov_b32 s61, 0
	s_mov_b32 s28, s27
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=1
	s_and_b64 vcc, exec, s[30:31]
	s_mov_b32 s28, s63
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_19
.LBB1_11:                               ; %.preheader510.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_13 Depth 2
	s_mov_b32 s29, s27
	s_add_co_i32 s63, s28, 1
	s_mul_u64 s[34:35], s[28:29], 0x88
	s_lshl_b32 s29, s28, 1
	s_cmp_eq_u32 s63, s33
	s_mov_b32 s64, 0
	s_cselect_b64 s[30:31], -1, 0
	s_add_nc_u64 s[34:35], s[48:49], s[34:35]
	s_mov_b64 s[58:59], 0
	s_mov_b64 s[56:57], -1
	s_branch .LBB1_13
.LBB1_12:                               ;   in Loop: Header=BB1_13 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v70, v84, v68
	v_mul_f32_e32 v71, v82, v68
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v69, v69
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v72, v80, v68
	v_cvt_f32_i32_e32 v30, v30
	v_mul_f32_e32 v28, v70, v28
	v_mul_f32_e32 v29, v71, v29
	v_mul_f32_e32 v70, v85, v68
	v_mul_f32_e32 v71, v83, v68
	v_cvt_f32_i32_e32 v31, v31
	v_mul_f32_e32 v30, v72, v30
	v_cvt_f32_i32_e32 v24, v24
	v_fmac_f32_e32 v28, v70, v69
	v_fmac_f32_e32 v29, v71, v69
	v_mul_f32_e32 v70, v78, v68
	v_mul_f32_e32 v71, v81, v68
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v132, v132, v28
	v_add_f32_e32 v133, v133, v29
	v_mul_f32_e32 v28, v70, v31
	v_mul_f32_e32 v29, v79, v68
	v_fmac_f32_e32 v30, v71, v69
	s_wait_dscnt 0x6
	v_mul_f32_e32 v31, v84, v66
	v_mul_f32_e32 v70, v82, v66
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v69
	v_add_f32_e32 v41, v41, v30
	v_cvt_f32_i32_e32 v29, v67
	v_mul_f32_e32 v24, v31, v24
	v_mul_f32_e32 v25, v70, v25
	v_mul_f32_e32 v30, v85, v66
	v_mul_f32_e32 v31, v83, v66
	v_add_f32_e32 v131, v131, v28
	v_mul_f32_e32 v28, v80, v66
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v24, v30, v29
	v_fmac_f32_e32 v25, v31, v29
	v_mul_f32_e32 v30, v78, v66
	v_mul_f32_e32 v26, v28, v26
	v_mul_f32_e32 v28, v81, v66
	v_add_f32_e32 v129, v129, v24
	v_add_f32_e32 v130, v130, v25
	v_mul_f32_e32 v24, v30, v27
	v_mul_f32_e32 v25, v79, v66
	v_fmac_f32_e32 v26, v28, v29
	s_wait_dscnt 0x5
	v_mul_f32_e32 v27, v84, v64
	v_mul_f32_e32 v28, v82, v64
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v25, v29
	v_add_f32_e32 v127, v127, v26
	v_cvt_f32_i32_e32 v25, v65
	v_mul_f32_e32 v20, v27, v20
	v_mul_f32_e32 v21, v28, v21
	v_mul_f32_e32 v26, v80, v64
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v27, v85, v64
	v_mul_f32_e32 v28, v83, v64
	v_mul_f32_e32 v30, v78, v64
	v_cvt_f32_i32_e32 v23, v23
	v_mul_f32_e32 v22, v26, v22
	v_mul_f32_e32 v26, v81, v64
	v_fmac_f32_e32 v20, v27, v25
	v_fmac_f32_e32 v21, v28, v25
	v_mul_f32_e32 v23, v30, v23
	v_mul_f32_e32 v27, v79, v64
	v_fmac_f32_e32 v22, v26, v25
	v_add_f32_e32 v126, v126, v20
	v_add_f32_e32 v125, v125, v21
	s_wait_dscnt 0x4
	v_mul_f32_e32 v20, v84, v54
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v21, v82, v54
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v23, v27, v25
	v_add_f32_e32 v124, v124, v22
	v_cvt_f32_i32_e32 v22, v55
	v_mul_f32_e32 v16, v20, v16
	v_mul_f32_e32 v20, v85, v54
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v80, v54
	v_cvt_f32_i32_e32 v18, v18
	v_add_f32_e32 v123, v123, v23
	v_mul_f32_e32 v23, v83, v54
	v_fmac_f32_e32 v16, v20, v22
	v_mul_f32_e32 v20, v78, v54
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v18, v21, v18
	v_mul_f32_e32 v21, v81, v54
	v_fmac_f32_e32 v17, v23, v22
	v_add_f32_e32 v122, v122, v16
	v_mul_f32_e32 v16, v20, v19
	v_mul_f32_e32 v19, v79, v54
	v_fmac_f32_e32 v18, v21, v22
	s_wait_dscnt 0x3
	v_mul_f32_e32 v20, v68, v62
	s_wait_dscnt 0x2
	v_mul_f32_e32 v21, v68, v60
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v121, v121, v17
	v_fmac_f32_e32 v16, v19, v22
	v_add_f32_e32 v119, v119, v18
	v_mul_f32_e32 v12, v20, v12
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v17, v68, v63
	v_mul_f32_e32 v18, v68, v61
	s_wait_dscnt 0x1
	v_mul_f32_e32 v19, v68, v56
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v120, v120, v16
	v_fmac_f32_e32 v12, v17, v69
	v_fmac_f32_e32 v13, v18, v69
	v_mul_f32_e32 v16, v68, v57
	v_mul_f32_e32 v14, v19, v14
	v_mul_f32_e32 v18, v66, v62
	v_mul_f32_e32 v19, v66, v60
	v_cvt_f32_i32_e32 v8, v8
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v117, v117, v12
	v_fmac_f32_e32 v14, v16, v69
	v_mul_f32_e32 v12, v66, v63
	v_mul_f32_e32 v8, v18, v8
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v16, v66, v61
	v_add_f32_e32 v118, v118, v13
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v8, v12, v29
	v_mul_f32_e32 v12, v66, v56
	v_fmac_f32_e32 v9, v16, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v13, v66, v58
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v113, v113, v8
	v_mul_f32_e32 v8, v12, v10
	v_add_f32_e32 v114, v114, v9
	v_mul_f32_e32 v9, v66, v57
	v_mul_f32_e32 v10, v13, v11
	v_mul_f32_e32 v11, v64, v62
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v8, v9, v29
	v_mul_f32_e32 v9, v64, v60
	v_mul_f32_e32 v12, v66, v59
	v_mul_f32_e32 v4, v11, v4
	v_mul_f32_e32 v11, v64, v63
	v_add_f32_e32 v112, v112, v8
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v8, v64, v61
	v_mul_f32_e32 v9, v64, v56
	v_fmac_f32_e32 v4, v11, v25
	v_mul_f32_e32 v11, v64, v58
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_fmac_f32_e32 v10, v12, v29
	v_fmac_f32_e32 v5, v8, v25
	v_add_f32_e32 v109, v109, v4
	v_mul_f32_e32 v4, v9, v6
	v_mul_f32_e32 v6, v11, v7
	v_mul_f32_e32 v7, v64, v57
	v_mul_f32_e32 v8, v64, v59
	v_mul_f32_e32 v20, v68, v58
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v111, v111, v10
	v_mul_f32_e32 v9, v54, v62
	v_mul_f32_e32 v10, v54, v60
	v_cvt_f32_i32_e32 v0, v0
	v_cvt_f32_i32_e32 v1, v1
	v_fmac_f32_e32 v4, v7, v25
	v_fmac_f32_e32 v6, v8, v25
	v_mul_f32_e32 v7, v54, v56
	v_mul_f32_e32 v8, v54, v58
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v17, v68, v59
	v_add_f32_e32 v110, v110, v5
	v_mul_f32_e32 v0, v9, v0
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v5, v54, v63
	v_mul_f32_e32 v9, v54, v61
	v_mul_f32_e32 v2, v7, v2
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v7, v54, v57
	v_mul_f32_e32 v8, v54, v59
	v_fmac_f32_e32 v15, v17, v69
	v_fmac_f32_e32 v0, v5, v22
	v_fmac_f32_e32 v1, v9, v22
	v_fmac_f32_e32 v2, v7, v22
	v_fmac_f32_e32 v3, v8, v22
	v_add_f32_e32 v128, v128, v24
	v_add_f32_e32 v115, v115, v14
	v_add_f32_e32 v116, v116, v15
	v_add_f32_e32 v107, v107, v4
	v_add_f32_e32 v108, v108, v6
	v_add_f32_e32 v103, v103, v0
	v_add_f32_e32 v106, v106, v1
	v_add_f32_e32 v105, v105, v2
	v_add_f32_e32 v104, v104, v3
	s_xor_b64 s[58:59], s[56:57], -1
	s_mov_b32 s64, 1
	s_mov_b64 s[56:57], 0
	s_and_b64 vcc, exec, s[58:59]
	s_mov_b64 s[58:59], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
.LBB1_13:                               ;   Parent Loop BB1_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s64, s29
	s_lshl_b32 s66, s64, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[68:69], s[26:27], s[42:43]
	s_mov_b32 s67, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[68:69], s[68:69], 0x48
	s_add_nc_u64 s[66:67], s[34:35], s[66:67]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[68:69], s[50:51], s[68:69]
	v_add_nc_u32_e32 v58, s60, v89
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[70:71], s68, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s69, 0, s[70:71]
	v_add_co_u32 v2, s[70:71], s66, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s67, 0, s[70:71]
	v_add_co_u32 v4, s[70:71], s68, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s69, 0, s[70:71]
	v_add_co_u32 v6, s[68:69], s66, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s67, 0, s[68:69]
	global_load_b64 v[76:77], v[0:1], off offset:40
	global_load_b64 v[70:71], v[2:3], off offset:40
	global_load_b64 v[74:75], v[4:5], off offset:40
	global_load_b64 v[72:73], v[6:7], off offset:40
	v_add_nc_u32_e32 v59, s61, v88
	ds_load_2addr_stride64_b32 v[54:55], v58 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v59 offset1:2
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:4 offset1:6
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v57, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v57, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[54:55], v58 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[58:59], v59 offset0:5 offset1:7
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v56, 0x2000, v101
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v76, s[16:17]
	v_add_nc_u32_e32 v55, 0x4000, v101
	v_cndmask_b32_e64 v57, 0, v77, s[16:17]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v56, v70, v71 offset1:16
	ds_store_2addr_b32 v55, v54, v57 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v74, s[18:19]
	v_cndmask_b32_e64 v55, 0, v75, s[18:19]
	v_add_nc_u32_e32 v56, 0x4000, v102
	v_add_nc_u32_e32 v57, 0x2000, v102
	ds_store_2addr_b32 v56, v54, v55 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v57, v72, v73 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[60:61], s[58:59], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[60:61]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_15
; %bb.14:                               ;   in Loop: Header=BB1_13 Depth=2
	s_add_co_i32 s26, s26, 1
	s_xor_b32 s72, s64, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[66:67], s[26:27], s[42:43]
	s_add_co_i32 s26, s64, s28
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[66:67], s[66:67], 0x48
	s_mul_u64 s[68:69], s[26:27], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[66:67], s[50:51], s[66:67]
	s_add_nc_u64 s[64:65], s[48:49], s[68:69]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[58:59], null, 0x48, v93, s[66:67]
	v_add_co_u32 v54, s[70:71], s66, v36
	s_lshl_b32 s68, s72, 6
	s_mov_b32 s69, s27
	s_mulk_i32 s26, 0x88
	v_add_co_ci_u32_e64 v55, null, s67, 0, s[70:71]
	v_add_co_u32 v58, vcc, v58, v90
	v_add_co_u32 v56, s[70:71], s66, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[64:65], s[64:65], s[68:69]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, 0, v59, vcc
	v_add_co_u32 v64, vcc, v32, s26
	v_add_co_ci_u32_e64 v57, null, s67, 0, s[70:71]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v60, s[66:67], s64, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s65, 0, s[66:67]
	v_add_co_u32 v62, s[66:67], s64, v39
	s_lshl_b32 s26, s72, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s65, 0, s[66:67]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v64, vcc, v64, s26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v65, vcc
	s_clause 0x1
	global_load_b64 v[76:77], v[54:55], off offset:8
	global_load_b64 v[74:75], v[56:57], off offset:8
	s_clause 0x1
	global_load_b64 v[70:71], v[60:61], off offset:8
	global_load_b64 v[72:73], v[62:63], off offset:8
	global_load_b32 v135, v[58:59], off
	global_load_b32 v136, v[64:65], off
.LBB1_15:                               ; %.preheader508.i.i
                                        ;   in Loop: Header=BB1_13 Depth=2
	v_add_nc_u32_e32 v60, 0, v89
	v_add_nc_u32_e32 v61, 0, v88
	s_xor_b64 s[60:61], s[60:61], -1
	s_and_b64 s[64:65], s[56:57], exec
	s_cselect_b32 s26, s41, 0x3400
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[58:59], v61 offset0:68 offset1:70
	s_cselect_b32 s64, s62, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[58:59], v61 offset0:69 offset1:71
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v58, s64, v94
	v_add_nc_u32_e32 v54, s26, v134
	s_and_not1_b64 vcc, exec, s[60:61]
	s_movk_i32 s60, 0x2000
	s_movk_i32 s61, 0x4000
	ds_load_2addr_b32 v[84:85], v58 offset1:1
	ds_load_2addr_b32 v[82:83], v58 offset0:2 offset1:3
	ds_load_2addr_b32 v[80:81], v58 offset0:4 offset1:5
	ds_load_2addr_b32 v[78:79], v58 offset0:6 offset1:7
	ds_load_2addr_b32 v[68:69], v54 offset1:1
	ds_load_2addr_b32 v[66:67], v54 offset0:32 offset1:33
	ds_load_2addr_b32 v[64:65], v54 offset0:64 offset1:65
	ds_load_2addr_b32 v[54:55], v54 offset0:96 offset1:97
	ds_load_2addr_b32 v[62:63], v58 offset0:32 offset1:33
	ds_load_2addr_b32 v[60:61], v58 offset0:34 offset1:35
	ds_load_2addr_b32 v[56:57], v58 offset0:36 offset1:37
	ds_load_2addr_b32 v[58:59], v58 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_12
; %bb.16:                               ; %.preheader509.i.i
                                        ;   in Loop: Header=BB1_13 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v137, v92, v136
	s_and_b64 s[58:59], s[58:59], exec
	v_cndmask_b32_e64 v76, 0, v76, s[16:17]
	v_cndmask_b32_e64 v77, 0, v77, s[16:17]
	s_cselect_b32 s26, s62, 0x3c00
	v_cvt_f32_f16_e64 v137, v137.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v138, s26, v91
	ds_store_b32 v101, v70 offset:4096
	ds_store_2addr_b32 v101, v76, v77 offset1:16
	v_cndmask_b32_e64 v70, 0, v74, s[18:19]
	v_cndmask_b32_e64 v74, 0, v75, s[18:19]
	v_cndmask_b32_e64 v137, 0, v137, s[22:23]
	s_cselect_b32 s26, s41, 0x3400
	s_mov_b32 s61, 0
	s_movk_i32 s60, 0x1000
	v_cndmask_b32_e64 v75, 0, v135, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s26, v91
	ds_store_b32 v101, v71 offset:4160
	ds_store_b32 v102, v72 offset:4096
	ds_store_2addr_b32 v102, v70, v74 offset1:16
	ds_store_b32 v102, v73 offset:4160
	ds_store_b32 v76, v75
	ds_store_b32 v138, v137
	s_branch .LBB1_12
.LBB1_17:
	s_branch .LBB1_178
.LBB1_18:                               ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB1_19:                               ; %._crit_edge580.i.i
	v_mad_u32_u24 v0, 0x50, v98, v97
	v_cmp_gt_i32_e64 s[26:27], s42, v87
	s_mov_b64 s[28:29], -1
	ds_store_2addr_b32 v0, v132, v133 offset1:20
	ds_store_2addr_b32 v0, v41, v131 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mad_co_i64_i32 v[0:1], null, s40, v87, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc, s38, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v86
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s39, v1, vcc
	s_and_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_592
; %bb.20:                               ; %Flow2275
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_595
.LBB1_21:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_596
.LBB1_22:                               ; %Flow2273
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_599
.LBB1_23:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_600
.LBB1_24:                               ; %Flow2271
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_603
.LBB1_25:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_604
.LBB1_26:                               ; %Flow2269
	v_mul_u32_u24_e32 v3, 0x50, v98
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_28
.LBB1_27:
	s_movk_i32 s28, 0x180
	buffer_load_b32 v4, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v4
	buffer_store_b32 v2, v33, s[44:47], s28 offen
.LBB1_28:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v41, v97, v3
	v_or_b32_e32 v5, 16, v87
	s_mov_b64 s[30:31], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[28:29], s42, v5
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v129, v130 offset1:20
	ds_store_2addr_b32 v41, v127, v128 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, vcc, s38, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s39, v3, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_607
; %bb.29:                               ; %Flow2267
	s_and_not1_b64 vcc, exec, s[30:31]
	s_lshl_b32 s41, s40, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_610
.LBB1_30:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_611
.LBB1_31:                               ; %Flow2265
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_614
.LBB1_32:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_615
.LBB1_33:                               ; %Flow2263
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_618
.LBB1_34:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_619
.LBB1_35:                               ; %Flow2261
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_37
.LBB1_36:
	s_add_co_i32 s30, s41, 0x180
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
.LBB1_37:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 32, v87
	s_mov_b64 s[34:35], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[30:31], s42, v7
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v126, v125 offset1:20
	ds_store_2addr_b32 v41, v124, v123 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, vcc, s38, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s39, v5, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_622
; %bb.38:                               ; %Flow2259
	s_and_not1_b64 vcc, exec, s[34:35]
	s_lshl_b32 s58, s40, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_625
.LBB1_39:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_626
.LBB1_40:                               ; %Flow2257
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_629
.LBB1_41:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_630
.LBB1_42:                               ; %Flow2255
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_633
.LBB1_43:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_634
.LBB1_44:                               ; %Flow2253
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_46
.LBB1_45:
	s_add_co_i32 s34, s58, 0x180
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
.LBB1_46:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 48, v87
	s_mov_b64 s[56:57], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[34:35], s42, v9
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v122, v121 offset1:20
	ds_store_2addr_b32 v41, v119, v120 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, vcc, s38, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s39, v7, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_637
; %bb.47:                               ; %Flow2251
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_640
.LBB1_48:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_641
.LBB1_49:                               ; %Flow2249
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_644
.LBB1_50:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_645
.LBB1_51:                               ; %Flow2247
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_648
.LBB1_52:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_649
.LBB1_53:                               ; %Flow2245
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_55
.LBB1_54:
	s_mul_i32 s56, s40, 0xc0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x180
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
.LBB1_55:                               ; %.preheader505.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v117, v118 offset1:20
	ds_store_2addr_b32 v41, v115, v116 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_652
; %bb.56:                               ; %Flow2243
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_655
.LBB1_57:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_656
.LBB1_58:                               ; %Flow2241
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_659
.LBB1_59:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_660
.LBB1_60:                               ; %Flow2239
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_663
.LBB1_61:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_664
.LBB1_62:                               ; %Flow2237
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_64
.LBB1_63:
	s_movk_i32 s26, 0x1c0
	buffer_load_b32 v0, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v8, v0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_64:                               ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v113, v114 offset1:20
	ds_store_2addr_b32 v41, v112, v111 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_667
; %bb.65:                               ; %Flow2235
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_670
.LBB1_66:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_671
.LBB1_67:                               ; %Flow2233
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_674
.LBB1_68:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_675
.LBB1_69:                               ; %Flow2231
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_678
.LBB1_70:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_679
.LBB1_71:                               ; %Flow2229
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_73
.LBB1_72:
	s_addk_co_i32 s41, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s41 offen
.LBB1_73:                               ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v109, v110 offset1:20
	ds_store_2addr_b32 v41, v107, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_682
; %bb.74:                               ; %Flow2227
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_685
.LBB1_75:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_686
.LBB1_76:                               ; %Flow2225
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_689
.LBB1_77:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_690
.LBB1_78:                               ; %Flow2223
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_693
.LBB1_79:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_694
.LBB1_80:                               ; %Flow2221
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_82
.LBB1_81:
	s_addk_co_i32 s58, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s58 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s58 offen
.LBB1_82:                               ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v103, v106 offset1:20
	ds_store_2addr_b32 v41, v105, v104 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_697
; %bb.83:                               ; %Flow2219
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_700
.LBB1_84:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_701
.LBB1_85:                               ; %Flow2217
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_704
.LBB1_86:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_705
.LBB1_87:                               ; %Flow2215
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_708
.LBB1_88:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_709
.LBB1_89:                               ; %Flow2213
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_91
.LBB1_90:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_91:                               ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb1ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b64 s[26:27], s[24:25]
	s_cbranch_execz .LBB1_95
; %bb.92:                               ; %.lr.ph.i.1.i
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[28:29], s[20:21]
	s_cbranch_execz .LBB1_94
; %bb.93:
	global_load_b32 v0, v[52:53], off
.LBB1_94:                               ; %.preheader511.loopexit.i.1.i
	s_or_b64 exec, exec, s[28:29]
	global_load_b32 v1, v[50:51], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v92, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[22:23]
	ds_store_2addr_stride64_b32 v91, v0, v1 offset0:48 offset1:56
.LBB1_95:                               ; %Flow2211
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	global_load_b64 v[0:1], v[42:43], off offset:8
	global_load_b64 v[2:3], v[46:47], off offset:8
	global_load_b64 v[4:5], v[44:45], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s27, 0
	s_and_not1_b64 vcc, exec, s[52:53]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[16:17]
	v_cndmask_b32_e64 v1, 0, v1, s[16:17]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[18:19]
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	ds_store_b32 v101, v2 offset:4096
	ds_store_2addr_b32 v101, v0, v1 offset1:16
	ds_store_b32 v101, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v102, v6 offset:4096
	ds_store_2addr_b32 v102, v4, v5 offset1:16
	ds_store_b32 v102, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_104
; %bb.96:                               ; %.preheader510.lr.ph.i.1.i
	v_or_b32_e32 v135, 0xf00, v95
	v_lshl_add_u32 v136, v96, 3, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	s_movk_i32 s60, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s62, 0x3800
	s_mov_b32 s61, 0
	s_mov_b32 s28, s27
	s_branch .LBB1_98
.LBB1_97:                               ;   in Loop: Header=BB1_98 Depth=1
	s_and_not1_b64 vcc, exec, s[30:31]
	s_mov_b32 s28, s63
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_104
.LBB1_98:                               ; %.preheader510.i.1.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_100 Depth 2
	s_mov_b32 s29, s27
	s_add_co_i32 s63, s28, 1
	s_mul_u64 s[34:35], s[28:29], 0x88
	s_lshl_b32 s29, s28, 1
	s_cmp_eq_u32 s63, s33
	s_mov_b32 s64, 0
	s_cselect_b64 s[30:31], -1, 0
	s_add_nc_u64 s[34:35], s[48:49], s[34:35]
	s_mov_b64 s[58:59], 0
	s_mov_b64 s[56:57], -1
	s_branch .LBB1_100
.LBB1_99:                               ;   in Loop: Header=BB1_100 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v70, v84, v68
	v_cvt_f32_i32_e32 v28, v28
	v_mul_f32_e32 v71, v82, v68
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v69, v69
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v72, v83, v68
	v_mul_f32_e32 v28, v70, v28
	v_mul_f32_e32 v70, v85, v68
	v_mul_f32_e32 v29, v71, v29
	v_mul_f32_e32 v71, v78, v68
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v28, v70, v69
	v_fmac_f32_e32 v29, v72, v69
	v_mul_f32_e32 v70, v80, v68
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v134, v134, v28
	v_add_f32_e32 v133, v133, v29
	v_mul_f32_e32 v28, v71, v30
	v_mul_f32_e32 v29, v79, v68
	v_mul_f32_e32 v30, v70, v31
	s_wait_dscnt 0x6
	v_mul_f32_e32 v70, v84, v66
	v_mul_f32_e32 v31, v81, v68
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v69
	v_cvt_f32_i32_e32 v29, v67
	v_mul_f32_e32 v67, v82, v66
	v_mul_f32_e32 v24, v70, v24
	v_mul_f32_e32 v70, v85, v66
	v_fmac_f32_e32 v30, v31, v69
	v_add_f32_e32 v132, v132, v28
	v_mul_f32_e32 v25, v67, v25
	v_mul_f32_e32 v28, v83, v66
	v_fmac_f32_e32 v24, v70, v29
	v_mul_f32_e32 v31, v78, v66
	v_mul_f32_e32 v67, v80, v66
	v_cvt_f32_i32_e32 v27, v27
	v_add_f32_e32 v131, v131, v30
	v_fmac_f32_e32 v25, v28, v29
	v_add_f32_e32 v130, v130, v24
	v_mul_f32_e32 v24, v31, v26
	v_mul_f32_e32 v26, v67, v27
	v_mul_f32_e32 v27, v79, v66
	v_mul_f32_e32 v28, v81, v66
	s_wait_dscnt 0x5
	v_mul_f32_e32 v30, v84, v64
	v_mul_f32_e32 v31, v82, v64
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v27, v29
	v_fmac_f32_e32 v26, v28, v29
	v_cvt_f32_i32_e32 v27, v65
	v_mul_f32_e32 v20, v30, v20
	v_mul_f32_e32 v21, v31, v21
	v_mul_f32_e32 v28, v85, v64
	v_mul_f32_e32 v30, v83, v64
	v_add_f32_e32 v128, v128, v24
	v_mul_f32_e32 v24, v78, v64
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v20, v28, v27
	v_fmac_f32_e32 v21, v30, v27
	v_add_f32_e32 v129, v129, v25
	v_mul_f32_e32 v25, v80, v64
	v_cvt_f32_i32_e32 v23, v23
	v_add_f32_e32 v126, v126, v20
	v_add_f32_e32 v125, v125, v21
	v_mul_f32_e32 v20, v24, v22
	v_mul_f32_e32 v21, v79, v64
	s_wait_dscnt 0x4
	v_mul_f32_e32 v24, v84, v54
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v22, v25, v23
	v_mul_f32_e32 v23, v81, v64
	v_fmac_f32_e32 v20, v21, v27
	v_cvt_f32_i32_e32 v21, v55
	v_mul_f32_e32 v25, v82, v54
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v16, v24, v16
	v_mul_f32_e32 v24, v85, v54
	v_fmac_f32_e32 v22, v23, v27
	v_add_f32_e32 v124, v124, v20
	v_mul_f32_e32 v17, v25, v17
	v_mul_f32_e32 v20, v83, v54
	v_fmac_f32_e32 v16, v24, v21
	v_mul_f32_e32 v23, v78, v54
	v_mul_f32_e32 v24, v80, v54
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v123, v123, v22
	v_fmac_f32_e32 v17, v20, v21
	v_add_f32_e32 v122, v122, v16
	v_mul_f32_e32 v16, v23, v18
	v_mul_f32_e32 v18, v24, v19
	v_mul_f32_e32 v19, v79, v54
	s_wait_dscnt 0x3
	v_mul_f32_e32 v22, v68, v60
	s_wait_dscnt 0x2
	v_mul_f32_e32 v23, v68, v62
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v121, v121, v17
	v_fmac_f32_e32 v16, v19, v21
	v_mul_f32_e32 v17, v68, v61
	v_mul_f32_e32 v12, v22, v12
	v_mul_f32_e32 v13, v23, v13
	v_mul_f32_e32 v19, v68, v63
	v_mul_f32_e32 v20, v81, v54
	v_cvt_f32_i32_e32 v14, v14
	v_fmac_f32_e32 v12, v17, v69
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v13, v19, v69
	v_fmac_f32_e32 v18, v20, v21
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v68, v58
	v_add_f32_e32 v118, v118, v12
	v_mul_f32_e32 v12, v66, v60
	v_add_f32_e32 v117, v117, v13
	v_mul_f32_e32 v13, v66, v62
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v120, v120, v16
	v_mul_f32_e32 v14, v20, v14
	v_mul_f32_e32 v16, v68, v59
	v_mul_f32_e32 v8, v12, v8
	v_mul_f32_e32 v12, v66, v61
	v_mul_f32_e32 v9, v13, v9
	v_mul_f32_e32 v13, v66, v58
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v14, v16, v69
	v_fmac_f32_e32 v8, v12, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v12, v66, v56
	v_cvt_f32_i32_e32 v11, v11
	v_mul_f32_e32 v10, v13, v10
	v_mul_f32_e32 v13, v66, v59
	v_add_f32_e32 v116, v116, v14
	v_mul_f32_e32 v14, v66, v63
	v_add_f32_e32 v114, v114, v8
	v_mul_f32_e32 v8, v12, v11
	v_mul_f32_e32 v11, v66, v57
	v_fmac_f32_e32 v10, v13, v29
	v_mul_f32_e32 v12, v64, v60
	v_cvt_f32_i32_e32 v4, v4
	v_fmac_f32_e32 v9, v14, v29
	v_fmac_f32_e32 v8, v11, v29
	v_add_f32_e32 v112, v112, v10
	v_mul_f32_e32 v10, v64, v61
	v_mul_f32_e32 v4, v12, v4
	v_mul_f32_e32 v11, v64, v58
	v_mul_f32_e32 v12, v64, v56
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_add_f32_e32 v113, v113, v9
	v_mul_f32_e32 v9, v64, v62
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v4, v10, v27
	v_mul_f32_e32 v6, v11, v6
	v_mul_f32_e32 v7, v12, v7
	v_mul_f32_e32 v10, v64, v59
	v_mul_f32_e32 v11, v64, v57
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v9, v64, v63
	v_add_f32_e32 v110, v110, v4
	v_fmac_f32_e32 v6, v10, v27
	v_fmac_f32_e32 v7, v11, v27
	v_mul_f32_e32 v4, v54, v60
	v_cvt_f32_i32_e32 v0, v0
	v_mul_f32_e32 v22, v68, v56
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v111, v111, v8
	v_fmac_f32_e32 v5, v9, v27
	v_mul_f32_e32 v8, v54, v62
	v_cvt_f32_i32_e32 v1, v1
	v_add_f32_e32 v108, v108, v6
	v_add_f32_e32 v107, v107, v7
	v_mul_f32_e32 v0, v4, v0
	v_mul_f32_e32 v4, v54, v61
	v_mul_f32_e32 v6, v54, v58
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v7, v54, v56
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v22, v15
	v_mul_f32_e32 v17, v68, v57
	v_add_f32_e32 v109, v109, v5
	v_mul_f32_e32 v1, v8, v1
	v_mul_f32_e32 v5, v54, v63
	v_mul_f32_e32 v2, v6, v2
	v_mul_f32_e32 v6, v54, v59
	v_fmac_f32_e32 v0, v4, v21
	v_mul_f32_e32 v3, v7, v3
	v_mul_f32_e32 v4, v54, v57
	v_fmac_f32_e32 v15, v17, v69
	v_fmac_f32_e32 v1, v5, v21
	v_fmac_f32_e32 v2, v6, v21
	v_add_f32_e32 v127, v127, v26
	v_fmac_f32_e32 v3, v4, v21
	v_add_f32_e32 v119, v119, v18
	v_add_f32_e32 v115, v115, v15
	v_add_f32_e32 v103, v103, v0
	v_add_f32_e32 v106, v106, v1
	v_add_f32_e32 v105, v105, v2
	v_add_f32_e32 v104, v104, v3
	s_xor_b64 s[58:59], s[56:57], -1
	s_mov_b32 s64, 1
	s_mov_b64 s[56:57], 0
	s_and_not1_b64 vcc, exec, s[58:59]
	s_mov_b64 s[58:59], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_97
.LBB1_100:                              ;   Parent Loop BB1_98 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s64, s29
	s_lshl_b32 s66, s64, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[68:69], s[26:27], s[42:43]
	s_mov_b32 s67, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[68:69], s[68:69], 0x48
	s_add_nc_u64 s[66:67], s[34:35], s[66:67]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[68:69], s[50:51], s[68:69]
	v_add_nc_u32_e32 v58, s60, v89
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[70:71], s68, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s69, 0, s[70:71]
	v_add_co_u32 v2, s[70:71], s66, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s67, 0, s[70:71]
	v_add_co_u32 v4, s[70:71], s68, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s69, 0, s[70:71]
	v_add_co_u32 v6, s[68:69], s66, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s67, 0, s[68:69]
	global_load_b64 v[76:77], v[0:1], off offset:40
	global_load_b64 v[70:71], v[2:3], off offset:40
	global_load_b64 v[74:75], v[4:5], off offset:40
	global_load_b64 v[72:73], v[6:7], off offset:40
	v_add_nc_u32_e32 v59, s61, v88
	ds_load_2addr_stride64_b32 v[54:55], v58 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v59 offset0:8 offset1:10
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:12 offset1:14
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v57, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v57, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[54:55], v58 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:9 offset1:11
	v_add_nc_u32_e32 v58, s61, v135
	ds_load_b32 v59, v59 offset:3328
	ds_load_b32 v58, v58
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v59, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v59, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v58, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v58, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v56, 0x2000, v101
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v76, s[16:17]
	v_add_nc_u32_e32 v55, 0x4000, v101
	v_cndmask_b32_e64 v57, 0, v77, s[16:17]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v56, v70, v71 offset1:16
	ds_store_2addr_b32 v55, v54, v57 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v74, s[18:19]
	v_cndmask_b32_e64 v55, 0, v75, s[18:19]
	v_add_nc_u32_e32 v56, 0x4000, v102
	v_add_nc_u32_e32 v57, 0x2000, v102
	ds_store_2addr_b32 v56, v54, v55 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v57, v72, v73 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[60:61], s[58:59], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[60:61]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_102
; %bb.101:                              ;   in Loop: Header=BB1_100 Depth=2
	s_add_co_i32 s26, s26, 1
	s_xor_b32 s72, s64, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[66:67], s[26:27], s[42:43]
	s_add_co_i32 s26, s64, s28
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[66:67], s[66:67], 0x48
	s_mul_u64 s[68:69], s[26:27], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[66:67], s[50:51], s[66:67]
	s_add_nc_u64 s[64:65], s[48:49], s[68:69]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[58:59], null, 0x48, v93, s[66:67]
	v_add_co_u32 v54, s[70:71], s66, v36
	s_lshl_b32 s68, s72, 6
	s_mov_b32 s69, s27
	s_mulk_i32 s26, 0x88
	v_add_co_ci_u32_e64 v55, null, s67, 0, s[70:71]
	v_add_co_u32 v58, vcc, v58, v90
	v_add_co_u32 v56, s[70:71], s66, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[64:65], s[64:65], s[68:69]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, 0, v59, vcc
	v_add_co_u32 v64, vcc, v32, s26
	v_add_co_ci_u32_e64 v57, null, s67, 0, s[70:71]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v60, s[66:67], s64, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s65, 0, s[66:67]
	v_add_co_u32 v62, s[66:67], s64, v39
	s_lshl_b32 s26, s72, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s65, 0, s[66:67]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v64, vcc, v64, s26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v65, vcc
	s_clause 0x1
	global_load_b64 v[76:77], v[54:55], off offset:8
	global_load_b64 v[74:75], v[56:57], off offset:8
	s_clause 0x1
	global_load_b64 v[70:71], v[60:61], off offset:8
	global_load_b64 v[72:73], v[62:63], off offset:8
	global_load_b32 v137, v[58:59], off
	global_load_b32 v138, v[64:65], off
.LBB1_102:                              ; %.preheader508.i.1.i
                                        ;   in Loop: Header=BB1_100 Depth=2
	v_add_nc_u32_e32 v60, 0, v89
	v_add_nc_u32_e32 v61, 0, v88
	s_xor_b64 s[60:61], s[60:61], -1
	s_and_b64 s[64:65], s[56:57], exec
	s_cselect_b32 s26, s41, 0x3400
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:72 offset1:74
	ds_load_2addr_stride64_b32 v[58:59], v61 offset0:76 offset1:78
	s_cselect_b32 s64, s62, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v58, 0, v135
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:73 offset1:75
	ds_load_b32 v59, v61 offset:19712
	ds_load_b32 v58, v58 offset:16384
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v59, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v59, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v58, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v58, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v56, s64, v94
	v_add_nc_u32_e32 v54, s26, v136
	s_and_not1_b64 vcc, exec, s[60:61]
	s_movk_i32 s60, 0x2000
	s_movk_i32 s61, 0x4000
	ds_load_2addr_b32 v[84:85], v56 offset1:1
	ds_load_2addr_b32 v[82:83], v56 offset0:2 offset1:3
	ds_load_2addr_b32 v[78:79], v56 offset0:4 offset1:5
	ds_load_2addr_b32 v[80:81], v56 offset0:6 offset1:7
	ds_load_2addr_b32 v[68:69], v54 offset0:128 offset1:129
	ds_load_2addr_b32 v[66:67], v54 offset0:160 offset1:161
	ds_load_2addr_b32 v[64:65], v54 offset0:192 offset1:193
	ds_load_2addr_b32 v[54:55], v54 offset0:224 offset1:225
	ds_load_2addr_b32 v[60:61], v56 offset0:32 offset1:33
	ds_load_2addr_b32 v[62:63], v56 offset0:34 offset1:35
	ds_load_2addr_b32 v[58:59], v56 offset0:36 offset1:37
	ds_load_2addr_b32 v[56:57], v56 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_99
; %bb.103:                              ; %.preheader509.i.1.i
                                        ;   in Loop: Header=BB1_100 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v139, v92, v138
	s_and_b64 s[58:59], s[58:59], exec
	v_cndmask_b32_e64 v76, 0, v76, s[16:17]
	v_cndmask_b32_e64 v77, 0, v77, s[16:17]
	s_cselect_b32 s26, s62, 0x3c00
	v_cvt_f32_f16_e64 v139, v139.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v140, s26, v91
	ds_store_b32 v101, v70 offset:4096
	ds_store_2addr_b32 v101, v76, v77 offset1:16
	v_cndmask_b32_e64 v70, 0, v74, s[18:19]
	v_cndmask_b32_e64 v74, 0, v75, s[18:19]
	v_cndmask_b32_e64 v139, 0, v139, s[22:23]
	s_cselect_b32 s26, s41, 0x3400
	s_mov_b32 s61, 0
	s_movk_i32 s60, 0x1000
	v_cndmask_b32_e64 v75, 0, v137, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s26, v91
	ds_store_b32 v101, v71 offset:4160
	ds_store_b32 v102, v72 offset:4096
	ds_store_2addr_b32 v102, v70, v74 offset1:16
	ds_store_b32 v102, v73 offset:4160
	ds_store_b32 v76, v75
	ds_store_b32 v140, v139
	s_branch .LBB1_99
.LBB1_104:                              ; %._crit_edge580.i.1.i
	ds_store_2addr_b32 v41, v134, v133 offset1:20
	ds_store_2addr_b32 v41, v132, v131 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v3, 64, v87
	s_mov_b64 s[28:29], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[0:1], null, s40, v3, 0
	v_cmp_gt_i32_e64 s[26:27], s42, v3
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v86
	v_add_co_u32 v0, vcc, s38, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s39, v1, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_712
; %bb.105:                              ; %Flow2206
	s_and_not1_b64 vcc, exec, s[28:29]
	s_lshl_b32 s41, s40, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_715
.LBB1_106:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_716
.LBB1_107:                              ; %Flow2204
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_719
.LBB1_108:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_720
.LBB1_109:                              ; %Flow2202
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_723
.LBB1_110:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_724
.LBB1_111:                              ; %Flow2200
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_113
.LBB1_112:
	s_add_co_i32 s28, s41, 0x180
	buffer_load_b32 v3, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s28 offen
.LBB1_113:                              ; %.preheader.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, 0x50, v87
	s_mov_b64 s[30:31], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[28:29], s42, v5
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v130, v129 offset1:20
	ds_store_2addr_b32 v41, v128, v127 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, vcc, s38, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s39, v3, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_727
; %bb.114:                              ; %Flow2198
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_730
.LBB1_115:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_731
.LBB1_116:                              ; %Flow2196
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_734
.LBB1_117:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_735
.LBB1_118:                              ; %Flow2194
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_738
.LBB1_119:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_739
.LBB1_120:                              ; %Flow2192
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_122
.LBB1_121:
	s_mul_i32 s30, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s30, 0x180
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
.LBB1_122:                              ; %.preheader.2.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 0x60, v87
	s_mov_b64 s[34:35], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[30:31], s42, v7
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v126, v125 offset1:20
	ds_store_2addr_b32 v41, v124, v123 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, vcc, s38, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s39, v5, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_742
; %bb.123:                              ; %Flow2190
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_745
.LBB1_124:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_746
.LBB1_125:                              ; %Flow2188
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_749
.LBB1_126:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_750
.LBB1_127:                              ; %Flow2186
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_753
.LBB1_128:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_754
.LBB1_129:                              ; %Flow2184
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_131
.LBB1_130:
	s_mul_i32 s34, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s34, 0x180
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
.LBB1_131:                              ; %.preheader.3.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 0x70, v87
	s_mov_b64 s[56:57], -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[34:35], s42, v9
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v122, v121 offset1:20
	ds_store_2addr_b32 v41, v120, v119 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, vcc, s38, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s39, v7, vcc
	s_and_not1_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_757
; %bb.132:                              ; %Flow2182
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_760
.LBB1_133:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_761
.LBB1_134:                              ; %Flow2180
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_764
.LBB1_135:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_765
.LBB1_136:                              ; %Flow2178
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_768
.LBB1_137:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_769
.LBB1_138:                              ; %Flow2176
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_140
.LBB1_139:
	s_mul_i32 s56, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x180
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
.LBB1_140:                              ; %.preheader505.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v118, v117 offset1:20
	ds_store_2addr_b32 v41, v116, v115 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_772
; %bb.141:                              ; %Flow2174
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_775
.LBB1_142:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_776
.LBB1_143:                              ; %Flow2172
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_779
.LBB1_144:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_780
.LBB1_145:                              ; %Flow2170
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_783
.LBB1_146:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_784
.LBB1_147:                              ; %Flow2168
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_149
.LBB1_148:
	s_addk_co_i32 s41, 0x1c0
	buffer_load_b32 v0, v33, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v8, v0
	buffer_store_b32 v0, v33, s[44:47], s41 offen
.LBB1_149:                              ; %.preheader.1.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v114, v113 offset1:20
	ds_store_2addr_b32 v41, v112, v111 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_787
; %bb.150:                              ; %Flow2166
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_790
.LBB1_151:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_791
.LBB1_152:                              ; %Flow2164
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_794
.LBB1_153:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_795
.LBB1_154:                              ; %Flow2162
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_798
.LBB1_155:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_799
.LBB1_156:                              ; %Flow2160
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_158
.LBB1_157:
	s_mul_i32 s26, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_158:                              ; %.preheader.2.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v110, v109 offset1:20
	ds_store_2addr_b32 v41, v108, v107 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_802
; %bb.159:                              ; %Flow2158
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_805
.LBB1_160:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_806
.LBB1_161:                              ; %Flow2156
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_809
.LBB1_162:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_810
.LBB1_163:                              ; %Flow2154
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_813
.LBB1_164:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_814
.LBB1_165:                              ; %Flow2152
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_167
.LBB1_166:
	s_mul_i32 s26, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_167:                              ; %.preheader.3.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v41, v103, v106 offset1:20
	ds_store_2addr_b32 v41, v105, v104 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_817
; %bb.168:                              ; %Flow2150
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_820
.LBB1_169:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_821
.LBB1_170:                              ; %Flow2148
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_824
.LBB1_171:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_825
.LBB1_172:                              ; %Flow2146
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_828
.LBB1_173:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_829
.LBB1_174:                              ; %Flow2144
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_176
.LBB1_175:
	s_mul_i32 s26, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x1c0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_176:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB1_177:                              ; %Flow2282
	s_cbranch_execnz .LBB1_18
.LBB1_178:
	s_and_b64 vcc, exec, s[54:55]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_180
; %bb.179:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb0ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.1.critedge.i
	s_mov_b64 s[26:27], 0
.LBB1_180:                              ; %Flow2421
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_18
; %bb.181:                              ; %.preheader509.i.i23
	s_and_saveexec_b64 s[26:27], s[24:25]
	s_cbranch_execz .LBB1_185
; %bb.182:                              ; %.lr.ph.i.i147
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[28:29], s[20:21]
	s_cbranch_execz .LBB1_184
; %bb.183:
	global_load_b32 v0, v[52:53], off
.LBB1_184:                              ; %.preheader506.loopexit.i.i
	s_or_b64 exec, exec, s[28:29]
	global_load_b32 v1, v[50:51], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v92, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[22:23]
	ds_store_2addr_stride64_b32 v91, v0, v1 offset0:48 offset1:56
.LBB1_185:                              ; %Flow2420
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	global_load_b64 v[0:1], v[42:43], off offset:8
	global_load_b64 v[2:3], v[46:47], off offset:8
	global_load_b64 v[4:5], v[44:45], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_add_nc_u32_e32 v99, 0, v99
	v_add_nc_u32_e32 v100, 0, v100
	v_mov_b32_e32 v109, 0
	v_cndmask_b32_e64 v41, 0, 1, s[52:53]
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v101, 0
	s_mov_b32 s27, 0
	s_and_not1_b64 vcc, exec, s[52:53]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[16:17]
	v_cndmask_b32_e64 v1, 0, v1, s[16:17]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[18:19]
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	ds_store_b32 v99, v2 offset:4096
	ds_store_2addr_b32 v99, v0, v1 offset1:16
	ds_store_b32 v99, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v100, v6 offset:4096
	ds_store_2addr_b32 v100, v4, v5 offset1:16
	ds_store_b32 v100, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_194
; %bb.186:                              ; %.preheader505.lr.ph.i.i
	v_lshl_add_u32 v133, v96, 3, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	s_movk_i32 s56, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s58, 0x3800
	s_mov_b32 s57, 0
	s_mov_b32 s28, s27
	s_branch .LBB1_188
.LBB1_187:                              ;   in Loop: Header=BB1_188 Depth=1
	s_and_b64 vcc, exec, s[30:31]
	s_mov_b32 s28, s59
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_194
.LBB1_188:                              ; %.preheader505.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_190 Depth 2
	s_mov_b32 s29, s27
	s_add_co_i32 s59, s28, 1
	s_mul_u64 s[34:35], s[28:29], 0x88
	s_lshl_b32 s29, s28, 1
	s_cmp_eq_u32 s59, s33
	s_mov_b32 s60, 0
	s_cselect_b64 s[30:31], -1, 0
	s_add_nc_u64 s[34:35], s[48:49], s[34:35]
	s_mov_b64 s[54:55], 0
	s_mov_b64 s[52:53], -1
	s_branch .LBB1_190
.LBB1_189:                              ;   in Loop: Header=BB1_190 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v70, v84, v68
	v_mul_f32_e32 v71, v82, v68
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v69, v69
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v72, v80, v68
	v_cvt_f32_i32_e32 v30, v30
	v_mul_f32_e32 v28, v70, v28
	v_mul_f32_e32 v29, v71, v29
	v_mul_f32_e32 v70, v85, v68
	v_mul_f32_e32 v71, v83, v68
	v_cvt_f32_i32_e32 v31, v31
	v_mul_f32_e32 v30, v72, v30
	v_cvt_f32_i32_e32 v24, v24
	v_fmac_f32_e32 v28, v70, v69
	v_fmac_f32_e32 v29, v71, v69
	v_mul_f32_e32 v70, v78, v68
	v_mul_f32_e32 v71, v81, v68
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v131, v131, v28
	v_add_f32_e32 v132, v132, v29
	v_mul_f32_e32 v28, v70, v31
	v_mul_f32_e32 v29, v79, v68
	v_fmac_f32_e32 v30, v71, v69
	s_wait_dscnt 0x6
	v_mul_f32_e32 v31, v84, v66
	v_mul_f32_e32 v70, v82, v66
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v69
	v_add_f32_e32 v129, v129, v30
	v_cvt_f32_i32_e32 v29, v67
	v_mul_f32_e32 v24, v31, v24
	v_mul_f32_e32 v25, v70, v25
	v_mul_f32_e32 v30, v85, v66
	v_mul_f32_e32 v31, v83, v66
	v_add_f32_e32 v130, v130, v28
	v_mul_f32_e32 v28, v80, v66
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v24, v30, v29
	v_fmac_f32_e32 v25, v31, v29
	v_mul_f32_e32 v30, v78, v66
	v_mul_f32_e32 v26, v28, v26
	v_mul_f32_e32 v28, v81, v66
	v_add_f32_e32 v127, v127, v24
	v_add_f32_e32 v128, v128, v25
	v_mul_f32_e32 v24, v30, v27
	v_mul_f32_e32 v25, v79, v66
	v_fmac_f32_e32 v26, v28, v29
	s_wait_dscnt 0x5
	v_mul_f32_e32 v27, v84, v64
	v_mul_f32_e32 v28, v82, v64
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v25, v29
	v_add_f32_e32 v125, v125, v26
	v_cvt_f32_i32_e32 v25, v65
	v_mul_f32_e32 v20, v27, v20
	v_mul_f32_e32 v21, v28, v21
	v_mul_f32_e32 v26, v80, v64
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v27, v85, v64
	v_mul_f32_e32 v28, v83, v64
	v_mul_f32_e32 v30, v78, v64
	v_cvt_f32_i32_e32 v23, v23
	v_mul_f32_e32 v22, v26, v22
	v_mul_f32_e32 v26, v81, v64
	v_fmac_f32_e32 v20, v27, v25
	v_fmac_f32_e32 v21, v28, v25
	v_mul_f32_e32 v23, v30, v23
	v_mul_f32_e32 v27, v79, v64
	v_fmac_f32_e32 v22, v26, v25
	v_add_f32_e32 v124, v124, v20
	v_add_f32_e32 v123, v123, v21
	s_wait_dscnt 0x4
	v_mul_f32_e32 v20, v84, v54
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v21, v82, v54
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v23, v27, v25
	v_add_f32_e32 v121, v121, v22
	v_cvt_f32_i32_e32 v22, v55
	v_mul_f32_e32 v16, v20, v16
	v_mul_f32_e32 v20, v85, v54
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v80, v54
	v_cvt_f32_i32_e32 v18, v18
	v_add_f32_e32 v122, v122, v23
	v_mul_f32_e32 v23, v83, v54
	v_fmac_f32_e32 v16, v20, v22
	v_mul_f32_e32 v20, v78, v54
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v18, v21, v18
	v_mul_f32_e32 v21, v81, v54
	v_fmac_f32_e32 v17, v23, v22
	v_add_f32_e32 v119, v119, v16
	v_mul_f32_e32 v16, v20, v19
	v_mul_f32_e32 v19, v79, v54
	v_fmac_f32_e32 v18, v21, v22
	s_wait_dscnt 0x3
	v_mul_f32_e32 v20, v68, v62
	s_wait_dscnt 0x2
	v_mul_f32_e32 v21, v68, v60
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v120, v120, v17
	v_fmac_f32_e32 v16, v19, v22
	v_add_f32_e32 v117, v117, v18
	v_mul_f32_e32 v12, v20, v12
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v17, v68, v63
	v_mul_f32_e32 v18, v68, v61
	s_wait_dscnt 0x1
	v_mul_f32_e32 v19, v68, v56
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v118, v118, v16
	v_fmac_f32_e32 v12, v17, v69
	v_fmac_f32_e32 v13, v18, v69
	v_mul_f32_e32 v16, v68, v57
	v_mul_f32_e32 v14, v19, v14
	v_mul_f32_e32 v18, v66, v62
	v_mul_f32_e32 v19, v66, v60
	v_cvt_f32_i32_e32 v8, v8
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v115, v115, v12
	v_fmac_f32_e32 v14, v16, v69
	v_mul_f32_e32 v12, v66, v63
	v_mul_f32_e32 v8, v18, v8
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v16, v66, v61
	v_add_f32_e32 v116, v116, v13
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v8, v12, v29
	v_mul_f32_e32 v12, v66, v56
	v_fmac_f32_e32 v9, v16, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v13, v66, v58
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v111, v111, v8
	v_mul_f32_e32 v8, v12, v10
	v_add_f32_e32 v112, v112, v9
	v_mul_f32_e32 v9, v66, v57
	v_mul_f32_e32 v10, v13, v11
	v_mul_f32_e32 v11, v64, v62
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v8, v9, v29
	v_mul_f32_e32 v9, v64, v60
	v_mul_f32_e32 v12, v66, v59
	v_mul_f32_e32 v4, v11, v4
	v_mul_f32_e32 v11, v64, v63
	v_add_f32_e32 v110, v110, v8
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v8, v64, v61
	v_mul_f32_e32 v9, v64, v56
	v_fmac_f32_e32 v4, v11, v25
	v_mul_f32_e32 v11, v64, v58
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_fmac_f32_e32 v10, v12, v29
	v_fmac_f32_e32 v5, v8, v25
	v_add_f32_e32 v107, v107, v4
	v_mul_f32_e32 v4, v9, v6
	v_mul_f32_e32 v6, v11, v7
	v_mul_f32_e32 v7, v64, v57
	v_mul_f32_e32 v8, v64, v59
	v_mul_f32_e32 v20, v68, v58
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v109, v109, v10
	v_mul_f32_e32 v9, v54, v62
	v_mul_f32_e32 v10, v54, v60
	v_cvt_f32_i32_e32 v0, v0
	v_cvt_f32_i32_e32 v1, v1
	v_fmac_f32_e32 v4, v7, v25
	v_fmac_f32_e32 v6, v8, v25
	v_mul_f32_e32 v7, v54, v56
	v_mul_f32_e32 v8, v54, v58
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v17, v68, v59
	v_add_f32_e32 v108, v108, v5
	v_mul_f32_e32 v0, v9, v0
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v5, v54, v63
	v_mul_f32_e32 v9, v54, v61
	v_mul_f32_e32 v2, v7, v2
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v7, v54, v57
	v_mul_f32_e32 v8, v54, v59
	v_fmac_f32_e32 v15, v17, v69
	v_fmac_f32_e32 v0, v5, v22
	v_fmac_f32_e32 v1, v9, v22
	v_fmac_f32_e32 v2, v7, v22
	v_fmac_f32_e32 v3, v8, v22
	v_add_f32_e32 v126, v126, v24
	v_add_f32_e32 v113, v113, v14
	v_add_f32_e32 v114, v114, v15
	v_add_f32_e32 v105, v105, v4
	v_add_f32_e32 v106, v106, v6
	v_add_f32_e32 v101, v101, v0
	v_add_f32_e32 v104, v104, v1
	v_add_f32_e32 v103, v103, v2
	v_add_f32_e32 v102, v102, v3
	s_xor_b64 s[54:55], s[52:53], -1
	s_mov_b32 s60, 1
	s_mov_b64 s[52:53], 0
	s_and_b64 vcc, exec, s[54:55]
	s_mov_b64 s[54:55], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_187
.LBB1_190:                              ;   Parent Loop BB1_188 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s60, s29
	s_lshl_b32 s62, s60, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[64:65], s[26:27], s[42:43]
	s_mov_b32 s63, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[64:65], s[64:65], 0x48
	s_add_nc_u64 s[62:63], s[34:35], s[62:63]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[64:65], s[50:51], s[64:65]
	v_add_nc_u32_e32 v58, s56, v89
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[66:67], s64, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s65, 0, s[66:67]
	v_add_co_u32 v2, s[66:67], s62, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s63, 0, s[66:67]
	v_add_co_u32 v4, s[66:67], s64, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s65, 0, s[66:67]
	v_add_co_u32 v6, s[64:65], s62, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s63, 0, s[64:65]
	global_load_b64 v[76:77], v[0:1], off offset:40
	global_load_b64 v[70:71], v[2:3], off offset:40
	global_load_b64 v[74:75], v[4:5], off offset:40
	global_load_b64 v[72:73], v[6:7], off offset:40
	v_add_nc_u32_e32 v59, s57, v88
	ds_load_2addr_stride64_b32 v[54:55], v58 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v59 offset1:2
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:4 offset1:6
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v57, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v56, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v57, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[54:55], v58 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[58:59], v59 offset0:5 offset1:7
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v56, 0x2000, v99
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v76, s[16:17]
	v_add_nc_u32_e32 v55, 0x4000, v99
	v_cndmask_b32_e64 v57, 0, v77, s[16:17]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v56, v70, v71 offset1:16
	ds_store_2addr_b32 v55, v54, v57 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v54, 0, v74, s[18:19]
	v_cndmask_b32_e64 v55, 0, v75, s[18:19]
	v_add_nc_u32_e32 v56, 0x4000, v100
	v_add_nc_u32_e32 v57, 0x2000, v100
	ds_store_2addr_b32 v56, v54, v55 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v57, v72, v73 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[56:57], s[54:55], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[56:57]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_192
; %bb.191:                              ;   in Loop: Header=BB1_190 Depth=2
	s_add_co_i32 s26, s26, 1
	s_xor_b32 s68, s60, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[62:63], s[26:27], s[42:43]
	s_add_co_i32 s26, s60, s28
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[62:63], s[62:63], 0x48
	s_mul_u64 s[64:65], s[26:27], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[62:63], s[50:51], s[62:63]
	s_add_nc_u64 s[60:61], s[48:49], s[64:65]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[58:59], null, 0x48, v93, s[62:63]
	v_add_co_u32 v54, s[66:67], s62, v36
	s_lshl_b32 s64, s68, 6
	s_mov_b32 s65, s27
	s_mulk_i32 s26, 0x88
	v_add_co_ci_u32_e64 v55, null, s63, 0, s[66:67]
	v_add_co_u32 v58, vcc, v58, v90
	v_add_co_u32 v56, s[66:67], s62, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[60:61], s[60:61], s[64:65]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v59, null, 0, v59, vcc
	v_add_co_u32 v64, vcc, v32, s26
	v_add_co_ci_u32_e64 v57, null, s63, 0, s[66:67]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v60, s[62:63], s60, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v61, null, s61, 0, s[62:63]
	v_add_co_u32 v62, s[62:63], s60, v39
	s_lshl_b32 s26, s68, 2
	v_add_co_ci_u32_e64 v63, null, s61, 0, s[62:63]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v64, vcc, v64, s26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v65, null, 0, v65, vcc
	s_clause 0x1
	global_load_b64 v[76:77], v[54:55], off offset:8
	global_load_b64 v[74:75], v[56:57], off offset:8
	s_clause 0x1
	global_load_b64 v[70:71], v[60:61], off offset:8
	global_load_b64 v[72:73], v[62:63], off offset:8
	global_load_b32 v134, v[58:59], off
	global_load_b32 v135, v[64:65], off
.LBB1_192:                              ; %.preheader503.i.i
                                        ;   in Loop: Header=BB1_190 Depth=2
	v_add_nc_u32_e32 v60, 0, v89
	v_add_nc_u32_e32 v61, 0, v88
	s_xor_b64 s[56:57], s[56:57], -1
	s_and_b64 s[60:61], s[52:53], exec
	s_cselect_b32 s26, s41, 0x3400
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[58:59], v61 offset0:68 offset1:70
	s_cselect_b32 s60, s58, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[54:55], v60 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[56:57], v61 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[58:59], v61 offset0:69 offset1:71
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v54, v56, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v54, v57, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v54, v58, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v54, v59, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v55, v56, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v55, v57, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v55, v58, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v55, v59, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v58, s60, v94
	v_add_nc_u32_e32 v54, s26, v133
	s_and_not1_b64 vcc, exec, s[56:57]
	s_movk_i32 s56, 0x2000
	s_movk_i32 s57, 0x4000
	ds_load_2addr_b32 v[84:85], v58 offset1:1
	ds_load_2addr_b32 v[82:83], v58 offset0:2 offset1:3
	ds_load_2addr_b32 v[80:81], v58 offset0:4 offset1:5
	ds_load_2addr_b32 v[78:79], v58 offset0:6 offset1:7
	ds_load_2addr_b32 v[68:69], v54 offset1:1
	ds_load_2addr_b32 v[66:67], v54 offset0:32 offset1:33
	ds_load_2addr_b32 v[64:65], v54 offset0:64 offset1:65
	ds_load_2addr_b32 v[54:55], v54 offset0:96 offset1:97
	ds_load_2addr_b32 v[62:63], v58 offset0:32 offset1:33
	ds_load_2addr_b32 v[60:61], v58 offset0:34 offset1:35
	ds_load_2addr_b32 v[56:57], v58 offset0:36 offset1:37
	ds_load_2addr_b32 v[58:59], v58 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_189
; %bb.193:                              ; %.preheader504.i.i
                                        ;   in Loop: Header=BB1_190 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v136, v92, v135
	s_and_b64 s[54:55], s[54:55], exec
	v_cndmask_b32_e64 v76, 0, v76, s[16:17]
	v_cndmask_b32_e64 v77, 0, v77, s[16:17]
	s_cselect_b32 s26, s58, 0x3c00
	v_cvt_f32_f16_e64 v136, v136.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v137, s26, v91
	ds_store_b32 v99, v70 offset:4096
	ds_store_2addr_b32 v99, v76, v77 offset1:16
	v_cndmask_b32_e64 v70, 0, v74, s[18:19]
	v_cndmask_b32_e64 v74, 0, v75, s[18:19]
	v_cndmask_b32_e64 v136, 0, v136, s[22:23]
	s_cselect_b32 s26, s41, 0x3400
	s_mov_b32 s57, 0
	s_movk_i32 s56, 0x1000
	v_cndmask_b32_e64 v75, 0, v134, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s26, v91
	ds_store_b32 v99, v71 offset:4160
	ds_store_b32 v100, v72 offset:4096
	ds_store_2addr_b32 v100, v70, v74 offset1:16
	ds_store_b32 v100, v73 offset:4160
	ds_store_b32 v76, v75
	ds_store_b32 v137, v136
	s_branch .LBB1_189
.LBB1_194:                              ; %._crit_edge575.i.i
	v_mad_u32_u24 v0, 0x50, v98, v97
	v_cmp_gt_i32_e64 s[26:27], s42, v87
	s_mov_b64 s[28:29], -1
	ds_store_2addr_b32 v0, v131, v132 offset1:20
	ds_store_2addr_b32 v0, v129, v130 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mad_co_i64_i32 v[0:1], null, s40, v87, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc, s38, v0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v86
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s39, v1, vcc
	s_and_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_198
; %bb.195:
	s_and_b64 s[30:31], s[14:15], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_197
; %bb.196:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off
.LBB1_197:                              ; %Flow2414
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[28:29], 0
.LBB1_198:                              ; %Flow2415
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_200
; %bb.199:
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], null offen
.LBB1_200:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:1280
	v_cndmask_b32_e64 v73, 0, 1, s[36:37]
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_204
; %bb.201:
	s_and_b64 s[30:31], s[12:13], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_203
; %bb.202:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:128
.LBB1_203:                              ; %Flow2412
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[28:29], 0
.LBB1_204:                              ; %Flow2413
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_206
; %bb.205:
	s_movk_i32 s28, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s28 offen
.LBB1_206:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[28:29], -1
	s_cbranch_vccnz .LBB1_210
; %bb.207:
	s_and_b64 s[30:31], s[10:11], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_209
; %bb.208:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:256
.LBB1_209:                              ; %Flow2410
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[28:29], 0
.LBB1_210:                              ; %Flow2411
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_212
; %bb.211:
	s_movk_i32 s28, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s28 offen
.LBB1_212:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[28:29], -1
	s_cbranch_vccnz .LBB1_216
; %bb.213:
	s_and_b64 s[30:31], s[8:9], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_215
; %bb.214:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:384
.LBB1_215:                              ; %Flow2408
	s_or_b64 exec, exec, s[28:29]
	s_mov_b64 s[28:29], 0
.LBB1_216:                              ; %Flow2409
	v_mul_u32_u24_e32 v3, 0x50, v98
	s_and_not1_b64 vcc, exec, s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_218
; %bb.217:
	s_movk_i32 s28, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s28 offen
.LBB1_218:                              ; %.preheader.1.i.i34
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v74, v97, v3
	v_or_b32_e32 v5, 16, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[28:29], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v127, v128 offset1:20
	ds_store_2addr_b32 v74, v125, v126 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[30:31], s38, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s39, v3, s[30:31]
	s_mov_b64 s[30:31], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_222
; %bb.219:
	s_and_b64 s[34:35], s[14:15], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_221
; %bb.220:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off
.LBB1_221:                              ; %Flow2406
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB1_222:                              ; %Flow2407
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_lshl_b32 s41, s40, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_224
; %bb.223:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s41 offen
.LBB1_224:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB1_228
; %bb.225:
	s_and_b64 s[34:35], s[12:13], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_227
; %bb.226:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:128
.LBB1_227:                              ; %Flow2404
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB1_228:                              ; %Flow2405
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_230
; %bb.229:
	s_add_co_i32 s30, s41, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s30 offen
.LBB1_230:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB1_234
; %bb.231:
	s_and_b64 s[34:35], s[10:11], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_233
; %bb.232:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:256
.LBB1_233:                              ; %Flow2402
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB1_234:                              ; %Flow2403
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_236
; %bb.235:
	s_add_co_i32 s30, s41, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s30 offen
.LBB1_236:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB1_240
; %bb.237:
	s_and_b64 s[34:35], s[8:9], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_239
; %bb.238:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:384
.LBB1_239:                              ; %Flow2400
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB1_240:                              ; %Flow2401
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_242
; %bb.241:
	s_add_co_i32 s30, s41, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s30 offen
.LBB1_242:                              ; %.preheader.2.i.i35
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 32, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[30:31], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v124, v123 offset1:20
	ds_store_2addr_b32 v74, v121, v122 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[34:35], s38, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s39, v5, s[34:35]
	s_mov_b64 s[34:35], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_246
; %bb.243:
	s_and_b64 s[36:37], s[14:15], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB1_245
; %bb.244:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off
.LBB1_245:                              ; %Flow2398
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[34:35], 0
.LBB1_246:                              ; %Flow2399
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[34:35]
	s_lshl_b32 s52, s40, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_248
; %bb.247:
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s52 offen
.LBB1_248:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB1_252
; %bb.249:
	s_and_b64 s[36:37], s[12:13], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB1_251
; %bb.250:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:128
.LBB1_251:                              ; %Flow2396
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[34:35], 0
.LBB1_252:                              ; %Flow2397
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_254
; %bb.253:
	s_add_co_i32 s34, s52, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s34 offen
.LBB1_254:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB1_258
; %bb.255:
	s_and_b64 s[36:37], s[10:11], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB1_257
; %bb.256:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:256
.LBB1_257:                              ; %Flow2394
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[34:35], 0
.LBB1_258:                              ; %Flow2395
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_260
; %bb.259:
	s_add_co_i32 s34, s52, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s34 offen
.LBB1_260:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[34:35], -1
	s_cbranch_vccnz .LBB1_264
; %bb.261:
	s_and_b64 s[36:37], s[8:9], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB1_263
; %bb.262:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:384
.LBB1_263:                              ; %Flow2392
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[34:35]
	s_mov_b64 s[34:35], 0
.LBB1_264:                              ; %Flow2393
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_266
; %bb.265:
	s_add_co_i32 s34, s52, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s34 offen
.LBB1_266:                              ; %.preheader.3.i.i36
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 48, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[34:35], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v119, v120 offset1:20
	ds_store_2addr_b32 v74, v117, v118 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[36:37], s38, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s39, v7, s[36:37]
	s_mov_b64 s[36:37], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_270
; %bb.267:
	s_and_b64 s[54:55], s[14:15], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_269
; %bb.268:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB1_269:                              ; %Flow2390
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_270:                              ; %Flow2391
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_272
; %bb.271:
	s_mul_i32 s36, s40, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_272:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_276
; %bb.273:
	s_and_b64 s[54:55], s[12:13], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_275
; %bb.274:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:128
.LBB1_275:                              ; %Flow2388
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_276:                              ; %Flow2389
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_278
; %bb.277:
	s_mul_i32 s36, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s36, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_278:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_282
; %bb.279:
	s_and_b64 s[54:55], s[10:11], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_281
; %bb.280:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:256
.LBB1_281:                              ; %Flow2386
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_282:                              ; %Flow2387
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_284
; %bb.283:
	s_mul_i32 s36, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s36, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_284:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_288
; %bb.285:
	s_and_b64 s[54:55], s[8:9], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_287
; %bb.286:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:384
.LBB1_287:                              ; %Flow2384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_288:                              ; %Flow2385
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_290
; %bb.289:
	s_mul_i32 s36, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s36, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_290:                              ; %.preheader500.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v115, v116 offset1:20
	ds_store_2addr_b32 v74, v113, v114 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_294
; %bb.291:
	s_and_b64 s[54:55], s[6:7], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_293
; %bb.292:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:64
.LBB1_293:                              ; %Flow2382
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_294:                              ; %Flow2383
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_296
; %bb.295:
	s_mov_b32 s36, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_296:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_300
; %bb.297:
	s_and_b64 s[54:55], s[4:5], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_299
; %bb.298:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:192
.LBB1_299:                              ; %Flow2380
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_300:                              ; %Flow2381
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_302
; %bb.301:
	s_movk_i32 s36, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_302:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_306
; %bb.303:
	s_and_b64 s[54:55], s[2:3], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[36:37], s[54:55]
	s_cbranch_execz .LBB1_305
; %bb.304:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:320
.LBB1_305:                              ; %Flow2378
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[36:37]
	s_mov_b64 s[36:37], 0
.LBB1_306:                              ; %Flow2379
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_308
; %bb.307:
	s_movk_i32 s36, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s36 offen
.LBB1_308:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[36:37], -1
	s_cbranch_vccnz .LBB1_312
; %bb.309:
	s_and_b64 s[36:37], s[0:1], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[26:27], s[36:37]
	s_cbranch_execz .LBB1_311
; %bb.310:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v8, off offset:448
.LBB1_311:                              ; %Flow2376
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[36:37], 0
.LBB1_312:                              ; %Flow2377
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_314
; %bb.313:
	s_movk_i32 s26, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s26 offen
.LBB1_314:                              ; %.preheader.1.1.i.i37
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v111, v112 offset1:20
	ds_store_2addr_b32 v74, v110, v109 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_318
; %bb.315:
	s_and_b64 s[36:37], s[6:7], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[26:27], s[36:37]
	s_cbranch_execz .LBB1_317
; %bb.316:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:64
.LBB1_317:                              ; %Flow2374
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_318:                              ; %Flow2375
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_320
; %bb.319:
	s_add_co_i32 s26, s41, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_320:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_324
; %bb.321:
	s_and_b64 s[36:37], s[4:5], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[26:27], s[36:37]
	s_cbranch_execz .LBB1_323
; %bb.322:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:192
.LBB1_323:                              ; %Flow2372
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_324:                              ; %Flow2373
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_326
; %bb.325:
	s_add_co_i32 s26, s41, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_326:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_330
; %bb.327:
	s_and_b64 s[36:37], s[2:3], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[26:27], s[36:37]
	s_cbranch_execz .LBB1_329
; %bb.328:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:320
.LBB1_329:                              ; %Flow2370
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_330:                              ; %Flow2371
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_332
; %bb.331:
	s_add_co_i32 s26, s41, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_332:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_336
; %bb.333:
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_335
; %bb.334:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_335:                              ; %Flow2368
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_336:                              ; %Flow2369
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_338
; %bb.337:
	s_addk_co_i32 s41, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s41 offen
.LBB1_338:                              ; %.preheader.2.1.i.i38
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v107, v108 offset1:20
	ds_store_2addr_b32 v74, v105, v106 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_342
; %bb.339:
	s_and_b64 s[28:29], s[6:7], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_341
; %bb.340:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB1_341:                              ; %Flow2366
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_342:                              ; %Flow2367
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_344
; %bb.343:
	s_or_b32 s26, s52, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_344:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_348
; %bb.345:
	s_and_b64 s[28:29], s[4:5], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_347
; %bb.346:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB1_347:                              ; %Flow2364
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_348:                              ; %Flow2365
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_350
; %bb.349:
	s_add_co_i32 s26, s52, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_350:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_354
; %bb.351:
	s_and_b64 s[28:29], s[2:3], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_353
; %bb.352:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB1_353:                              ; %Flow2362
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_354:                              ; %Flow2363
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_356
; %bb.355:
	s_add_co_i32 s26, s52, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_356:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_360
; %bb.357:
	s_and_b64 s[28:29], s[0:1], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_359
; %bb.358:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_359:                              ; %Flow2360
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_360:                              ; %Flow2361
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_362
; %bb.361:
	s_addk_co_i32 s52, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s52 offen
.LBB1_362:                              ; %.preheader.3.1.i.i39
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v101, v104 offset1:20
	ds_store_2addr_b32 v74, v103, v102 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_366
; %bb.363:
	s_and_b64 s[28:29], s[6:7], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_365
; %bb.364:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB1_365:                              ; %Flow2358
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_366:                              ; %Flow2359
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_368
; %bb.367:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s26, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_368:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_372
; %bb.369:
	s_and_b64 s[28:29], s[4:5], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_371
; %bb.370:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB1_371:                              ; %Flow2356
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_372:                              ; %Flow2357
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_374
; %bb.373:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_374:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_378
; %bb.375:
	s_and_b64 s[28:29], s[2:3], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_377
; %bb.376:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB1_377:                              ; %Flow2354
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_378:                              ; %Flow2355
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_380
; %bb.379:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_380:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[26:27], -1
	s_cbranch_vccnz .LBB1_384
; %bb.381:
	s_and_b64 s[28:29], s[0:1], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_383
; %bb.382:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_383:                              ; %Flow2352
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_mov_b64 s[26:27], 0
.LBB1_384:                              ; %Flow2353
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_386
; %bb.385:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s26 offen
.LBB1_386:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb0ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b64 s[26:27], s[24:25]
	s_cbranch_execz .LBB1_390
; %bb.387:                              ; %.lr.ph.i.1.i102
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[24:25], s[20:21]
	s_cbranch_execz .LBB1_389
; %bb.388:
	global_load_b32 v0, v[52:53], off
.LBB1_389:                              ; %.preheader506.loopexit.i.1.i
	s_or_b64 exec, exec, s[24:25]
	global_load_b32 v1, v[50:51], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v92, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[22:23]
	ds_store_2addr_stride64_b32 v91, v0, v1 offset0:48 offset1:56
.LBB1_390:                              ; %Flow2351
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	global_load_b64 v[0:1], v[42:43], off offset:8
	global_load_b64 v[2:3], v[46:47], off offset:8
	global_load_b64 v[4:5], v[44:45], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_cmp_ne_u32_e32 vcc, 1, v41
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v75, 0
	s_mov_b32 s25, 0
	s_and_b64 vcc, exec, vcc
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[16:17]
	v_cndmask_b32_e64 v1, 0, v1, s[16:17]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[18:19]
	v_cndmask_b32_e64 v5, 0, v5, s[18:19]
	ds_store_b32 v99, v2 offset:4096
	ds_store_2addr_b32 v99, v0, v1 offset1:16
	ds_store_b32 v99, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v100, v6 offset:4096
	ds_store_2addr_b32 v100, v4, v5 offset1:16
	ds_store_b32 v100, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_399
; %bb.391:                              ; %.preheader505.lr.ph.i.1.i
	v_or_b32_e32 v95, 0xf00, v95
	v_lshl_add_u32 v96, v96, 3, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	s_movk_i32 s52, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s54, 0x3800
	s_mov_b32 s53, 0
	s_mov_b32 s26, s25
	s_branch .LBB1_393
.LBB1_392:                              ;   in Loop: Header=BB1_393 Depth=1
	s_and_not1_b64 vcc, exec, s[28:29]
	s_mov_b32 s26, s55
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_399
.LBB1_393:                              ; %.preheader505.i.1.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_395 Depth 2
	s_mov_b32 s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s26, 1
	s_mul_u64 s[30:31], s[26:27], 0x88
	s_lshl_b32 s27, s26, 1
	s_cmp_eq_u32 s55, s33
	s_mov_b32 s56, 0
	s_cselect_b64 s[28:29], -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[48:49], s[30:31]
	s_mov_b64 s[36:37], 0
	s_mov_b64 s[34:35], -1
	s_branch .LBB1_395
.LBB1_394:                              ;   in Loop: Header=BB1_395 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v57, v71, v55
	v_cvt_f32_i32_e32 v28, v28
	v_mul_f32_e32 v58, v69, v55
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v56, v56
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v59, v70, v55
	v_mul_f32_e32 v28, v57, v28
	v_mul_f32_e32 v57, v72, v55
	v_mul_f32_e32 v29, v58, v29
	v_mul_f32_e32 v58, v65, v55
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v28, v57, v56
	v_fmac_f32_e32 v29, v59, v56
	v_mul_f32_e32 v57, v67, v55
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v119, v119, v28
	v_add_f32_e32 v118, v118, v29
	v_mul_f32_e32 v28, v58, v30
	v_mul_f32_e32 v29, v66, v55
	v_mul_f32_e32 v30, v57, v31
	s_wait_dscnt 0x6
	v_mul_f32_e32 v57, v71, v53
	v_mul_f32_e32 v31, v68, v55
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v56
	v_cvt_f32_i32_e32 v29, v54
	v_mul_f32_e32 v54, v69, v53
	v_mul_f32_e32 v24, v57, v24
	v_mul_f32_e32 v57, v72, v53
	v_fmac_f32_e32 v30, v31, v56
	v_add_f32_e32 v117, v117, v28
	v_mul_f32_e32 v25, v54, v25
	v_mul_f32_e32 v28, v70, v53
	v_fmac_f32_e32 v24, v57, v29
	v_mul_f32_e32 v31, v65, v53
	v_mul_f32_e32 v54, v67, v53
	v_cvt_f32_i32_e32 v27, v27
	v_add_f32_e32 v116, v116, v30
	v_fmac_f32_e32 v25, v28, v29
	v_add_f32_e32 v115, v115, v24
	v_mul_f32_e32 v24, v31, v26
	v_mul_f32_e32 v26, v54, v27
	v_mul_f32_e32 v27, v66, v53
	v_mul_f32_e32 v28, v68, v53
	s_wait_dscnt 0x5
	v_mul_f32_e32 v30, v71, v51
	v_mul_f32_e32 v31, v69, v51
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v27, v29
	v_fmac_f32_e32 v26, v28, v29
	v_cvt_f32_i32_e32 v27, v52
	v_mul_f32_e32 v20, v30, v20
	v_mul_f32_e32 v21, v31, v21
	v_mul_f32_e32 v28, v72, v51
	v_mul_f32_e32 v30, v70, v51
	v_add_f32_e32 v113, v113, v24
	v_mul_f32_e32 v24, v65, v51
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v20, v28, v27
	v_fmac_f32_e32 v21, v30, v27
	v_add_f32_e32 v114, v114, v25
	v_mul_f32_e32 v25, v67, v51
	v_cvt_f32_i32_e32 v23, v23
	v_add_f32_e32 v111, v111, v20
	v_add_f32_e32 v110, v110, v21
	v_mul_f32_e32 v20, v24, v22
	v_mul_f32_e32 v21, v66, v51
	s_wait_dscnt 0x4
	v_mul_f32_e32 v24, v71, v41
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v22, v25, v23
	v_mul_f32_e32 v23, v68, v51
	v_fmac_f32_e32 v20, v21, v27
	v_cvt_f32_i32_e32 v21, v42
	v_mul_f32_e32 v25, v69, v41
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v16, v24, v16
	v_mul_f32_e32 v24, v72, v41
	v_fmac_f32_e32 v22, v23, v27
	v_add_f32_e32 v109, v109, v20
	v_mul_f32_e32 v17, v25, v17
	v_mul_f32_e32 v20, v70, v41
	v_fmac_f32_e32 v16, v24, v21
	v_mul_f32_e32 v23, v65, v41
	v_mul_f32_e32 v24, v67, v41
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v108, v108, v22
	v_fmac_f32_e32 v17, v20, v21
	v_add_f32_e32 v107, v107, v16
	v_mul_f32_e32 v16, v23, v18
	v_mul_f32_e32 v18, v24, v19
	v_mul_f32_e32 v19, v66, v41
	s_wait_dscnt 0x3
	v_mul_f32_e32 v22, v55, v47
	s_wait_dscnt 0x2
	v_mul_f32_e32 v23, v55, v49
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v106, v106, v17
	v_fmac_f32_e32 v16, v19, v21
	v_mul_f32_e32 v17, v55, v48
	v_mul_f32_e32 v12, v22, v12
	v_mul_f32_e32 v13, v23, v13
	v_mul_f32_e32 v19, v55, v50
	v_mul_f32_e32 v20, v68, v41
	v_cvt_f32_i32_e32 v14, v14
	v_fmac_f32_e32 v12, v17, v56
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v13, v19, v56
	v_fmac_f32_e32 v18, v20, v21
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v55, v45
	v_add_f32_e32 v103, v103, v12
	v_mul_f32_e32 v12, v53, v47
	v_add_f32_e32 v102, v102, v13
	v_mul_f32_e32 v13, v53, v49
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v105, v105, v16
	v_mul_f32_e32 v14, v20, v14
	v_mul_f32_e32 v16, v55, v46
	v_mul_f32_e32 v8, v12, v8
	v_mul_f32_e32 v12, v53, v48
	v_mul_f32_e32 v9, v13, v9
	v_mul_f32_e32 v13, v53, v45
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v14, v16, v56
	v_fmac_f32_e32 v8, v12, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v12, v53, v43
	v_cvt_f32_i32_e32 v11, v11
	v_mul_f32_e32 v10, v13, v10
	v_mul_f32_e32 v13, v53, v46
	v_add_f32_e32 v101, v101, v14
	v_mul_f32_e32 v14, v53, v50
	v_add_f32_e32 v97, v97, v8
	v_mul_f32_e32 v8, v12, v11
	v_mul_f32_e32 v11, v53, v44
	v_fmac_f32_e32 v10, v13, v29
	v_mul_f32_e32 v12, v51, v47
	v_cvt_f32_i32_e32 v4, v4
	v_fmac_f32_e32 v9, v14, v29
	v_fmac_f32_e32 v8, v11, v29
	v_add_f32_e32 v84, v84, v10
	v_mul_f32_e32 v10, v51, v48
	v_mul_f32_e32 v4, v12, v4
	v_mul_f32_e32 v11, v51, v45
	v_mul_f32_e32 v12, v51, v43
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_add_f32_e32 v85, v85, v9
	v_mul_f32_e32 v9, v51, v49
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v4, v10, v27
	v_mul_f32_e32 v6, v11, v6
	v_mul_f32_e32 v7, v12, v7
	v_mul_f32_e32 v10, v51, v46
	v_mul_f32_e32 v11, v51, v44
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v9, v51, v50
	v_add_f32_e32 v82, v82, v4
	v_fmac_f32_e32 v6, v10, v27
	v_fmac_f32_e32 v7, v11, v27
	v_mul_f32_e32 v4, v41, v47
	v_cvt_f32_i32_e32 v0, v0
	v_mul_f32_e32 v22, v55, v43
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v83, v83, v8
	v_fmac_f32_e32 v5, v9, v27
	v_mul_f32_e32 v8, v41, v49
	v_cvt_f32_i32_e32 v1, v1
	v_add_f32_e32 v80, v80, v6
	v_add_f32_e32 v79, v79, v7
	v_mul_f32_e32 v0, v4, v0
	v_mul_f32_e32 v4, v41, v48
	v_mul_f32_e32 v6, v41, v45
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v7, v41, v43
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v22, v15
	v_mul_f32_e32 v17, v55, v44
	v_add_f32_e32 v81, v81, v5
	v_mul_f32_e32 v1, v8, v1
	v_mul_f32_e32 v5, v41, v50
	v_mul_f32_e32 v2, v6, v2
	v_mul_f32_e32 v6, v41, v46
	v_fmac_f32_e32 v0, v4, v21
	v_mul_f32_e32 v3, v7, v3
	v_mul_f32_e32 v4, v41, v44
	v_fmac_f32_e32 v15, v17, v56
	v_fmac_f32_e32 v1, v5, v21
	v_fmac_f32_e32 v2, v6, v21
	v_add_f32_e32 v112, v112, v26
	v_fmac_f32_e32 v3, v4, v21
	v_add_f32_e32 v104, v104, v18
	v_add_f32_e32 v98, v98, v15
	v_add_f32_e32 v75, v75, v0
	v_add_f32_e32 v78, v78, v1
	v_add_f32_e32 v77, v77, v2
	v_add_f32_e32 v76, v76, v3
	s_xor_b64 s[36:37], s[34:35], -1
	s_mov_b32 s56, 1
	s_mov_b64 s[34:35], 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[36:37], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_392
.LBB1_395:                              ;   Parent Loop BB1_393 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s24, s56, s27
	s_lshl_b32 s58, s56, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[24:25], s[42:43]
	s_mov_b32 s59, s25
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[60:61], 0x48
	s_add_nc_u64 s[58:59], s[30:31], s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[60:61], s[50:51], s[60:61]
	v_add_nc_u32_e32 v45, s52, v89
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[62:63], s60, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s61, 0, s[62:63]
	v_add_co_u32 v2, s[62:63], s58, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s59, 0, s[62:63]
	v_add_co_u32 v4, s[62:63], s60, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s61, 0, s[62:63]
	v_add_co_u32 v6, s[60:61], s58, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s59, 0, s[60:61]
	global_load_b64 v[63:64], v[0:1], off offset:40
	global_load_b64 v[57:58], v[2:3], off offset:40
	global_load_b64 v[61:62], v[4:5], off offset:40
	global_load_b64 v[59:60], v[6:7], off offset:40
	v_add_nc_u32_e32 v46, s53, v88
	ds_load_2addr_stride64_b32 v[41:42], v45 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v46 offset0:8 offset1:10
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:12 offset1:14
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v44, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v44, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[41:42], v45 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:9 offset1:11
	v_add_nc_u32_e32 v45, s53, v95
	ds_load_b32 v46, v46 offset:3328
	ds_load_b32 v45, v45
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v43, 0x2000, v99
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v63, s[16:17]
	v_add_nc_u32_e32 v42, 0x4000, v99
	v_cndmask_b32_e64 v44, 0, v64, s[16:17]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v43, v57, v58 offset1:16
	ds_store_2addr_b32 v42, v41, v44 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v61, s[18:19]
	v_cndmask_b32_e64 v42, 0, v62, s[18:19]
	v_add_nc_u32_e32 v43, 0x4000, v100
	v_add_nc_u32_e32 v44, 0x2000, v100
	ds_store_2addr_b32 v43, v41, v42 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v44, v59, v60 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[52:53], s[36:37], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[52:53]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_397
; %bb.396:                              ;   in Loop: Header=BB1_395 Depth=2
	s_add_co_i32 s24, s24, 1
	s_xor_b32 s64, s56, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[24:25], s[42:43]
	s_add_co_i32 s24, s56, s26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[58:59], 0x48
	s_mul_u64 s[60:61], s[24:25], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[58:59], s[50:51], s[58:59]
	s_add_nc_u64 s[56:57], s[48:49], s[60:61]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[45:46], null, 0x48, v93, s[58:59]
	v_add_co_u32 v41, s[62:63], s58, v36
	s_lshl_b32 s60, s64, 6
	s_mov_b32 s61, s25
	s_mulk_i32 s24, 0x88
	v_add_co_ci_u32_e64 v42, null, s59, 0, s[62:63]
	v_add_co_u32 v45, vcc, v45, v90
	v_add_co_u32 v43, s[62:63], s58, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[56:57], s[56:57], s[60:61]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, 0, v46, vcc
	v_add_co_u32 v51, vcc, v32, s24
	v_add_co_ci_u32_e64 v44, null, s59, 0, s[62:63]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v47, s[58:59], s56, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v48, null, s57, 0, s[58:59]
	v_add_co_u32 v49, s[58:59], s56, v39
	s_lshl_b32 s24, s64, 2
	v_add_co_ci_u32_e64 v50, null, s57, 0, s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v51, vcc, v51, s24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v52, vcc
	s_clause 0x1
	global_load_b64 v[63:64], v[41:42], off offset:8
	global_load_b64 v[61:62], v[43:44], off offset:8
	s_clause 0x1
	global_load_b64 v[57:58], v[47:48], off offset:8
	global_load_b64 v[59:60], v[49:50], off offset:8
	global_load_b32 v120, v[45:46], off
	global_load_b32 v121, v[51:52], off
.LBB1_397:                              ; %.preheader503.i.1.i
                                        ;   in Loop: Header=BB1_395 Depth=2
	v_add_nc_u32_e32 v47, 0, v89
	v_add_nc_u32_e32 v48, 0, v88
	s_xor_b64 s[52:53], s[52:53], -1
	s_and_b64 s[56:57], s[34:35], exec
	s_cselect_b32 s24, s41, 0x3400
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:72 offset1:74
	ds_load_2addr_stride64_b32 v[45:46], v48 offset0:76 offset1:78
	s_cselect_b32 s56, s54, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v45, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v46, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v45, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v46, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v45, 0, v95
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:73 offset1:75
	ds_load_b32 v46, v48 offset:19712
	ds_load_b32 v45, v45 offset:16384
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v43, s56, v94
	v_add_nc_u32_e32 v41, s24, v96
	s_and_not1_b64 vcc, exec, s[52:53]
	s_movk_i32 s52, 0x2000
	s_movk_i32 s53, 0x4000
	ds_load_2addr_b32 v[71:72], v43 offset1:1
	ds_load_2addr_b32 v[69:70], v43 offset0:2 offset1:3
	ds_load_2addr_b32 v[65:66], v43 offset0:4 offset1:5
	ds_load_2addr_b32 v[67:68], v43 offset0:6 offset1:7
	ds_load_2addr_b32 v[55:56], v41 offset0:128 offset1:129
	ds_load_2addr_b32 v[53:54], v41 offset0:160 offset1:161
	ds_load_2addr_b32 v[51:52], v41 offset0:192 offset1:193
	ds_load_2addr_b32 v[41:42], v41 offset0:224 offset1:225
	ds_load_2addr_b32 v[47:48], v43 offset0:32 offset1:33
	ds_load_2addr_b32 v[49:50], v43 offset0:34 offset1:35
	ds_load_2addr_b32 v[45:46], v43 offset0:36 offset1:37
	ds_load_2addr_b32 v[43:44], v43 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_394
; %bb.398:                              ; %.preheader504.i.1.i
                                        ;   in Loop: Header=BB1_395 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v122, v92, v121
	s_and_b64 s[36:37], s[36:37], exec
	v_cndmask_b32_e64 v63, 0, v63, s[16:17]
	v_cndmask_b32_e64 v64, 0, v64, s[16:17]
	s_cselect_b32 s24, s54, 0x3c00
	v_cvt_f32_f16_e32 v122, v122.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v123, s24, v91
	ds_store_b32 v99, v57 offset:4096
	ds_store_2addr_b32 v99, v63, v64 offset1:16
	v_cndmask_b32_e64 v57, 0, v61, s[18:19]
	v_cndmask_b32_e64 v61, 0, v62, s[18:19]
	v_cndmask_b32_e64 v122, 0, v122, s[22:23]
	s_cselect_b32 s24, s41, 0x3400
	s_mov_b32 s53, 0
	s_movk_i32 s52, 0x1000
	v_cndmask_b32_e64 v62, 0, v120, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v63, s24, v91
	ds_store_b32 v99, v58 offset:4160
	ds_store_b32 v100, v59 offset:4096
	ds_store_2addr_b32 v100, v57, v61 offset1:16
	ds_store_b32 v100, v60 offset:4160
	ds_store_b32 v63, v62
	ds_store_b32 v123, v122
	s_branch .LBB1_394
.LBB1_399:                              ; %._crit_edge575.i.1.i
	ds_store_2addr_b32 v74, v119, v118 offset1:20
	ds_store_2addr_b32 v74, v117, v116 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v3, 64, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[0:1], null, s40, v3, 0
	v_cmp_gt_i32_e64 s[16:17], s42, v3
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v86
	v_add_co_u32 v0, s[18:19], s38, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s39, v1, s[18:19]
	s_mov_b64 s[18:19], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_403
; %bb.400:
	s_and_b64 s[20:21], s[14:15], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_402
; %bb.401:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off
.LBB1_402:                              ; %Flow2345
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_403:                              ; %Flow2346
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_lshl_b32 s26, s40, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_405
; %bb.404:
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s26 offen
.LBB1_405:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[18:19], -1
	s_cbranch_vccnz .LBB1_409
; %bb.406:
	s_and_b64 s[20:21], s[12:13], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_408
; %bb.407:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:128
.LBB1_408:                              ; %Flow2343
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_409:                              ; %Flow2344
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_411
; %bb.410:
	s_or_b32 s18, s26, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s18 offen
.LBB1_411:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[18:19], -1
	s_cbranch_vccnz .LBB1_415
; %bb.412:
	s_and_b64 s[20:21], s[10:11], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_414
; %bb.413:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:256
.LBB1_414:                              ; %Flow2341
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_415:                              ; %Flow2342
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_417
; %bb.416:
	s_add_co_i32 s18, s26, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s18 offen
.LBB1_417:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[18:19], -1
	s_cbranch_vccnz .LBB1_421
; %bb.418:
	s_and_b64 s[20:21], s[8:9], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB1_420
; %bb.419:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:384
.LBB1_420:                              ; %Flow2339
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB1_421:                              ; %Flow2340
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_423
; %bb.422:
	s_add_co_i32 s18, s26, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v33, s[44:47], s18 offen
.LBB1_423:                              ; %.preheader.1.i.1.i52
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, 0x50, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[18:19], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v115, v114 offset1:20
	ds_store_2addr_b32 v74, v113, v112 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[20:21], s38, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s39, v3, s[20:21]
	s_mov_b64 s[20:21], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_427
; %bb.424:
	s_and_b64 s[22:23], s[14:15], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB1_426
; %bb.425:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off
.LBB1_426:                              ; %Flow2337
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB1_427:                              ; %Flow2338
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_429
; %bb.428:
	s_mul_i32 s20, s40, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s20 offen
.LBB1_429:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB1_433
; %bb.430:
	s_and_b64 s[22:23], s[12:13], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB1_432
; %bb.431:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:128
.LBB1_432:                              ; %Flow2335
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB1_433:                              ; %Flow2336
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_435
; %bb.434:
	s_mul_i32 s20, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s20, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s20 offen
.LBB1_435:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB1_439
; %bb.436:
	s_and_b64 s[22:23], s[10:11], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB1_438
; %bb.437:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:256
.LBB1_438:                              ; %Flow2333
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB1_439:                              ; %Flow2334
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_441
; %bb.440:
	s_mul_i32 s20, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s20, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s20 offen
.LBB1_441:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[20:21], -1
	s_cbranch_vccnz .LBB1_445
; %bb.442:
	s_and_b64 s[22:23], s[8:9], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB1_444
; %bb.443:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:384
.LBB1_444:                              ; %Flow2331
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB1_445:                              ; %Flow2332
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_447
; %bb.446:
	s_mul_i32 s20, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s20, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v33, s[44:47], s20 offen
.LBB1_447:                              ; %.preheader.2.i.1.i53
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 0x60, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[20:21], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v111, v110 offset1:20
	ds_store_2addr_b32 v74, v109, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[22:23], s38, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s39, v5, s[22:23]
	s_mov_b64 s[22:23], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_451
; %bb.448:
	s_and_b64 s[24:25], s[14:15], s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[22:23], s[24:25]
	s_cbranch_execz .LBB1_450
; %bb.449:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off
.LBB1_450:                              ; %Flow2329
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB1_451:                              ; %Flow2330
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_453
; %bb.452:
	s_mul_i32 s22, s40, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s22 offen
.LBB1_453:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[22:23], -1
	s_cbranch_vccnz .LBB1_457
; %bb.454:
	s_and_b64 s[24:25], s[12:13], s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[22:23], s[24:25]
	s_cbranch_execz .LBB1_456
; %bb.455:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:128
.LBB1_456:                              ; %Flow2327
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB1_457:                              ; %Flow2328
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_459
; %bb.458:
	s_mul_i32 s22, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s22, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s22 offen
.LBB1_459:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[22:23], -1
	s_cbranch_vccnz .LBB1_463
; %bb.460:
	s_and_b64 s[24:25], s[10:11], s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[22:23], s[24:25]
	s_cbranch_execz .LBB1_462
; %bb.461:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:256
.LBB1_462:                              ; %Flow2325
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB1_463:                              ; %Flow2326
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_465
; %bb.464:
	s_mul_i32 s22, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s22, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s22 offen
.LBB1_465:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[22:23], -1
	s_cbranch_vccnz .LBB1_469
; %bb.466:
	s_and_b64 s[24:25], s[8:9], s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[22:23], s[24:25]
	s_cbranch_execz .LBB1_468
; %bb.467:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:384
.LBB1_468:                              ; %Flow2323
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB1_469:                              ; %Flow2324
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_471
; %bb.470:
	s_mul_i32 s22, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s22, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v33, s[44:47], s22 offen
.LBB1_471:                              ; %.preheader.3.i.1.i54
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 0x70, v87
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[22:23], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v107, v106 offset1:20
	ds_store_2addr_b32 v74, v105, v104 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[24:25], s38, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s39, v7, s[24:25]
	s_mov_b64 s[24:25], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_475
; %bb.472:
	s_and_b64 s[24:25], s[14:15], s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[14:15], s[24:25]
	s_cbranch_execz .LBB1_474
; %bb.473:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB1_474:                              ; %Flow2321
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[14:15]
	s_mov_b64 s[24:25], 0
.LBB1_475:                              ; %Flow2322
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[24:25]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_477
; %bb.476:
	s_mul_i32 s14, s40, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s14 offen
.LBB1_477:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[14:15], -1
	s_cbranch_vccnz .LBB1_481
; %bb.478:
	s_and_b64 s[14:15], s[12:13], s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB1_480
; %bb.479:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:128
.LBB1_480:                              ; %Flow2319
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[14:15], 0
.LBB1_481:                              ; %Flow2320
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_483
; %bb.482:
	s_mul_i32 s12, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s12, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s12 offen
.LBB1_483:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[12:13], -1
	s_cbranch_vccnz .LBB1_487
; %bb.484:
	s_and_b64 s[12:13], s[10:11], s[22:23]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[10:11], s[12:13]
	s_cbranch_execz .LBB1_486
; %bb.485:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:256
.LBB1_486:                              ; %Flow2317
	s_or_b64 exec, exec, s[10:11]
	s_mov_b64 s[12:13], 0
.LBB1_487:                              ; %Flow2318
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_489
; %bb.488:
	s_mul_i32 s10, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s10, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s10 offen
.LBB1_489:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[10:11], -1
	s_cbranch_vccnz .LBB1_493
; %bb.490:
	s_and_b64 s[10:11], s[8:9], s[22:23]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_492
; %bb.491:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:384
.LBB1_492:                              ; %Flow2315
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[10:11], 0
.LBB1_493:                              ; %Flow2316
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_495
; %bb.494:
	s_mul_i32 s8, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s8 offen
.LBB1_495:                              ; %.preheader500.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v103, v102 offset1:20
	ds_store_2addr_b32 v74, v101, v98 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_499
; %bb.496:
	s_and_b64 s[10:11], s[6:7], s[16:17]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_498
; %bb.497:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:64
.LBB1_498:                              ; %Flow2313
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_499:                              ; %Flow2314
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_501
; %bb.500:
	s_or_b32 s8, s26, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s8 offen
.LBB1_501:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_505
; %bb.502:
	s_and_b64 s[10:11], s[4:5], s[16:17]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_504
; %bb.503:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:192
.LBB1_504:                              ; %Flow2311
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_505:                              ; %Flow2312
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_507
; %bb.506:
	s_or_b32 s8, s26, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s8 offen
.LBB1_507:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_511
; %bb.508:
	s_and_b64 s[10:11], s[2:3], s[16:17]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_510
; %bb.509:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:320
.LBB1_510:                              ; %Flow2309
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_511:                              ; %Flow2310
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_513
; %bb.512:
	s_add_co_i32 s8, s26, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s8 offen
.LBB1_513:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_517
; %bb.514:
	s_and_b64 s[10:11], s[0:1], s[16:17]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_516
; %bb.515:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v8, off offset:448
.LBB1_516:                              ; %Flow2307
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_517:                              ; %Flow2308
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_519
; %bb.518:
	s_addk_co_i32 s26, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v33, s[44:47], s26 offen
.LBB1_519:                              ; %.preheader.1.1.i.1.i55
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v97, v85 offset1:20
	ds_store_2addr_b32 v74, v84, v83 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_523
; %bb.520:
	s_and_b64 s[10:11], s[6:7], s[18:19]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_522
; %bb.521:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:64
.LBB1_522:                              ; %Flow2305
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_523:                              ; %Flow2306
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_525
; %bb.524:
	s_mul_i32 s8, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s8, s8, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_525:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_529
; %bb.526:
	s_and_b64 s[10:11], s[4:5], s[18:19]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_528
; %bb.527:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:192
.LBB1_528:                              ; %Flow2303
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_529:                              ; %Flow2304
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_531
; %bb.530:
	s_mul_i32 s8, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_531:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_535
; %bb.532:
	s_and_b64 s[10:11], s[2:3], s[18:19]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_534
; %bb.533:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:320
.LBB1_534:                              ; %Flow2301
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_535:                              ; %Flow2302
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_537
; %bb.536:
	s_mul_i32 s8, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_537:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_541
; %bb.538:
	s_and_b64 s[10:11], s[0:1], s[18:19]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_540
; %bb.539:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_540:                              ; %Flow2299
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_541:                              ; %Flow2300
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_543
; %bb.542:
	s_mul_i32 s8, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_543:                              ; %.preheader.2.1.i.1.i56
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v82, v81 offset1:20
	ds_store_2addr_b32 v74, v80, v79 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_547
; %bb.544:
	s_and_b64 s[10:11], s[6:7], s[20:21]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_546
; %bb.545:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB1_546:                              ; %Flow2297
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_547:                              ; %Flow2298
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_549
; %bb.548:
	s_mul_i32 s8, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s8, s8, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_549:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_553
; %bb.550:
	s_and_b64 s[10:11], s[4:5], s[20:21]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_552
; %bb.551:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB1_552:                              ; %Flow2295
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_553:                              ; %Flow2296
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_555
; %bb.554:
	s_mul_i32 s8, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_555:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_559
; %bb.556:
	s_and_b64 s[10:11], s[2:3], s[20:21]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_558
; %bb.557:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB1_558:                              ; %Flow2293
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_559:                              ; %Flow2294
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_561
; %bb.560:
	s_mul_i32 s8, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_561:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB1_565
; %bb.562:
	s_and_b64 s[10:11], s[0:1], s[20:21]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB1_564
; %bb.563:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_564:                              ; %Flow2291
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB1_565:                              ; %Flow2292
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_567
; %bb.566:
	s_mul_i32 s8, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s8, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s8 offen
.LBB1_567:                              ; %.preheader.3.1.i.1.i57
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v74, v75, v78 offset1:20
	ds_store_2addr_b32 v74, v77, v76 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v86
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_571
; %bb.568:
	s_and_b64 s[8:9], s[6:7], s[22:23]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execz .LBB1_570
; %bb.569:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB1_570:                              ; %Flow2289
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[8:9], 0
.LBB1_571:                              ; %Flow2290
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_573
; %bb.572:
	s_mul_i32 s6, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s6, s6, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s6 offen
.LBB1_573:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB1_577
; %bb.574:
	s_and_b64 s[6:7], s[4:5], s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB1_576
; %bb.575:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB1_576:                              ; %Flow2287
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[6:7], 0
.LBB1_577:                              ; %Flow2288
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_579
; %bb.578:
	s_mul_i32 s4, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s4, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s4 offen
.LBB1_579:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB1_583
; %bb.580:
	s_and_b64 s[4:5], s[2:3], s[22:23]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB1_582
; %bb.581:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB1_582:                              ; %Flow2285
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[4:5], 0
.LBB1_583:                              ; %Flow2286
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_585
; %bb.584:
	s_mul_i32 s2, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s2, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s2 offen
.LBB1_585:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v86 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v73
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_589
; %bb.586:
	s_and_b64 s[2:3], s[0:1], s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_588
; %bb.587:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_588:                              ; %Flow2283
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_589:                              ; %Flow2284
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_591
; %bb.590:
	s_mul_i32 s0, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v33, s[44:47], s0 offen
.LBB1_591:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.LBB1_592:
	s_and_b64 s[30:31], s[14:15], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_594
; %bb.593:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off
.LBB1_594:                              ; %Flow2274
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_21
.LBB1_595:
	buffer_load_b32 v3, v33, s[44:47], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], null offen
	ds_load_b32 v2, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_22
.LBB1_596:
	s_and_b64 s[30:31], s[12:13], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_598
; %bb.597:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:128
.LBB1_598:                              ; %Flow2272
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_23
.LBB1_599:
	s_movk_i32 s28, 0x80
	buffer_load_b32 v3, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s28 offen
	ds_load_b32 v2, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_24
.LBB1_600:
	s_and_b64 s[30:31], s[10:11], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_602
; %bb.601:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:256
.LBB1_602:                              ; %Flow2270
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_25
.LBB1_603:
	s_movk_i32 s28, 0x100
	buffer_load_b32 v3, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s28 offen
	ds_load_b32 v2, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_26
.LBB1_604:
	s_and_b64 s[30:31], s[8:9], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_606
; %bb.605:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:384
.LBB1_606:                              ; %Flow2268
	s_or_b64 exec, exec, s[28:29]
	v_mul_u32_u24_e32 v3, 0x50, v98
	s_cbranch_execz .LBB1_27
	s_branch .LBB1_28
.LBB1_607:
	s_and_b64 s[34:35], s[14:15], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_609
; %bb.608:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off
.LBB1_609:                              ; %Flow2266
	s_or_b64 exec, exec, s[30:31]
	s_lshl_b32 s41, s40, 6
	s_cbranch_execnz .LBB1_30
.LBB1_610:
	buffer_load_b32 v5, v33, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s41 offen
	ds_load_b32 v4, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_31
.LBB1_611:
	s_and_b64 s[34:35], s[12:13], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_613
; %bb.612:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:128
.LBB1_613:                              ; %Flow2264
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_32
.LBB1_614:
	s_add_co_i32 s30, s41, 0x80
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
	ds_load_b32 v4, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_33
.LBB1_615:
	s_and_b64 s[34:35], s[10:11], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_617
; %bb.616:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:256
.LBB1_617:                              ; %Flow2262
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_34
.LBB1_618:
	s_add_co_i32 s30, s41, 0x100
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
	ds_load_b32 v4, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_35
.LBB1_619:
	s_and_b64 s[34:35], s[8:9], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_621
; %bb.620:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:384
.LBB1_621:                              ; %Flow2260
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB1_36
	s_branch .LBB1_37
.LBB1_622:
	s_and_b64 s[56:57], s[14:15], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_624
; %bb.623:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_624:                              ; %Flow2258
	s_or_b64 exec, exec, s[34:35]
	s_lshl_b32 s58, s40, 7
	s_cbranch_execnz .LBB1_39
.LBB1_625:
	buffer_load_b32 v7, v33, s[44:47], s58 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s58 offen
	ds_load_b32 v6, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_40
.LBB1_626:
	s_and_b64 s[56:57], s[12:13], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_628
; %bb.627:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB1_628:                              ; %Flow2256
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_41
.LBB1_629:
	s_add_co_i32 s34, s58, 0x80
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
	ds_load_b32 v6, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_42
.LBB1_630:
	s_and_b64 s[56:57], s[10:11], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_632
; %bb.631:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB1_632:                              ; %Flow2254
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_43
.LBB1_633:
	s_add_co_i32 s34, s58, 0x100
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
	ds_load_b32 v6, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_44
.LBB1_634:
	s_and_b64 s[56:57], s[8:9], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_636
; %bb.635:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB1_636:                              ; %Flow2252
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB1_45
	s_branch .LBB1_46
.LBB1_637:
	s_and_b64 s[60:61], s[14:15], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_639
; %bb.638:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB1_639:                              ; %Flow2250
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_48
.LBB1_640:
	s_mul_i32 s56, s40, 0xc0
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_49
.LBB1_641:
	s_and_b64 s[60:61], s[12:13], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_643
; %bb.642:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:128
.LBB1_643:                              ; %Flow2248
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_50
.LBB1_644:
	s_mul_i32 s56, s40, 0xc0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x80
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_51
.LBB1_645:
	s_and_b64 s[60:61], s[10:11], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_647
; %bb.646:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:256
.LBB1_647:                              ; %Flow2246
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_52
.LBB1_648:
	s_mul_i32 s56, s40, 0xc0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x100
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_53
.LBB1_649:
	s_and_b64 s[60:61], s[8:9], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_651
; %bb.650:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:384
.LBB1_651:                              ; %Flow2244
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execz .LBB1_54
	s_branch .LBB1_55
.LBB1_652:
	s_and_b64 s[60:61], s[6:7], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_654
; %bb.653:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:64
.LBB1_654:                              ; %Flow2242
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_57
.LBB1_655:
	s_mov_b32 s56, 64
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_58
.LBB1_656:
	s_and_b64 s[60:61], s[4:5], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_658
; %bb.657:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:192
.LBB1_658:                              ; %Flow2240
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_59
.LBB1_659:
	s_movk_i32 s56, 0xc0
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_60
.LBB1_660:
	s_and_b64 s[60:61], s[2:3], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[56:57], s[60:61]
	s_cbranch_execz .LBB1_662
; %bb.661:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:320
.LBB1_662:                              ; %Flow2238
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_61
.LBB1_663:
	s_movk_i32 s56, 0x140
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_62
.LBB1_664:
	s_and_b64 s[56:57], s[0:1], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_666
; %bb.665:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	global_load_b32 v9, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v8, v9
	global_store_b32 v[0:1], v9, off offset:448
.LBB1_666:                              ; %Flow2236
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_63
	s_branch .LBB1_64
.LBB1_667:
	s_and_b64 s[56:57], s[6:7], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_669
; %bb.668:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:64
.LBB1_669:                              ; %Flow2234
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_66
.LBB1_670:
	s_add_co_i32 s26, s41, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_67
.LBB1_671:
	s_and_b64 s[56:57], s[4:5], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_673
; %bb.672:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:192
.LBB1_673:                              ; %Flow2232
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_68
.LBB1_674:
	s_add_co_i32 s26, s41, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_69
.LBB1_675:
	s_and_b64 s[56:57], s[2:3], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_677
; %bb.676:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:320
.LBB1_677:                              ; %Flow2230
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_70
.LBB1_678:
	s_add_co_i32 s26, s41, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_71
.LBB1_679:
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_681
; %bb.680:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_681:                              ; %Flow2228
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_72
	s_branch .LBB1_73
.LBB1_682:
	s_and_b64 s[28:29], s[6:7], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_684
; %bb.683:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB1_684:                              ; %Flow2226
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_75
.LBB1_685:
	s_or_b32 s26, s58, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_76
.LBB1_686:
	s_and_b64 s[28:29], s[4:5], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_688
; %bb.687:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB1_688:                              ; %Flow2224
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_77
.LBB1_689:
	s_add_co_i32 s26, s58, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_78
.LBB1_690:
	s_and_b64 s[28:29], s[2:3], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_692
; %bb.691:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB1_692:                              ; %Flow2222
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_79
.LBB1_693:
	s_add_co_i32 s26, s58, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_80
.LBB1_694:
	s_and_b64 s[28:29], s[0:1], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_696
; %bb.695:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_696:                              ; %Flow2220
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_81
	s_branch .LBB1_82
.LBB1_697:
	s_and_b64 s[28:29], s[6:7], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_699
; %bb.698:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB1_699:                              ; %Flow2218
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_84
.LBB1_700:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s26, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_85
.LBB1_701:
	s_and_b64 s[28:29], s[4:5], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_703
; %bb.702:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB1_703:                              ; %Flow2216
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_86
.LBB1_704:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_87
.LBB1_705:
	s_and_b64 s[28:29], s[2:3], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_707
; %bb.706:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB1_707:                              ; %Flow2214
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_88
.LBB1_708:
	s_mul_i32 s26, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_89
.LBB1_709:
	s_and_b64 s[28:29], s[0:1], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_711
; %bb.710:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_711:                              ; %Flow2212
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_90
	s_branch .LBB1_91
.LBB1_712:
	s_and_b64 s[30:31], s[14:15], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_714
; %bb.713:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off
.LBB1_714:                              ; %Flow2205
	s_or_b64 exec, exec, s[28:29]
	s_lshl_b32 s41, s40, 8
	s_cbranch_execnz .LBB1_106
.LBB1_715:
	buffer_load_b32 v3, v33, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s41 offen
	ds_load_b32 v2, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_107
.LBB1_716:
	s_and_b64 s[30:31], s[12:13], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_718
; %bb.717:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:128
.LBB1_718:                              ; %Flow2203
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_108
.LBB1_719:
	s_or_b32 s28, s41, 0x80
	buffer_load_b32 v3, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s28 offen
	ds_load_b32 v2, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_109
.LBB1_720:
	s_and_b64 s[30:31], s[10:11], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_722
; %bb.721:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:256
.LBB1_722:                              ; %Flow2201
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execnz .LBB1_110
.LBB1_723:
	s_add_co_i32 s28, s41, 0x100
	buffer_load_b32 v3, v33, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v33, s[44:47], s28 offen
	ds_load_b32 v2, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[28:29], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_111
.LBB1_724:
	s_and_b64 s[30:31], s[8:9], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[28:29], s[30:31]
	s_cbranch_execz .LBB1_726
; %bb.725:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:384
.LBB1_726:                              ; %Flow2199
	s_or_b64 exec, exec, s[28:29]
	s_cbranch_execz .LBB1_112
	s_branch .LBB1_113
.LBB1_727:
	s_and_b64 s[34:35], s[14:15], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_729
; %bb.728:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off
.LBB1_729:                              ; %Flow2197
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_115
.LBB1_730:
	s_mul_i32 s30, s40, 0x140
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
	ds_load_b32 v4, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_116
.LBB1_731:
	s_and_b64 s[34:35], s[12:13], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_733
; %bb.732:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:128
.LBB1_733:                              ; %Flow2195
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_117
.LBB1_734:
	s_mul_i32 s30, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s30, 0x80
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
	ds_load_b32 v4, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_118
.LBB1_735:
	s_and_b64 s[34:35], s[10:11], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_737
; %bb.736:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:256
.LBB1_737:                              ; %Flow2193
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execnz .LBB1_119
.LBB1_738:
	s_mul_i32 s30, s40, 0x140
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s30, 0x100
	buffer_load_b32 v5, v33, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v33, s[44:47], s30 offen
	ds_load_b32 v4, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[30:31], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_120
.LBB1_739:
	s_and_b64 s[34:35], s[8:9], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB1_741
; %bb.740:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:384
.LBB1_741:                              ; %Flow2191
	s_or_b64 exec, exec, s[30:31]
	s_cbranch_execz .LBB1_121
	s_branch .LBB1_122
.LBB1_742:
	s_and_b64 s[56:57], s[14:15], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_744
; %bb.743:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_744:                              ; %Flow2189
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_124
.LBB1_745:
	s_mul_i32 s34, s40, 0x180
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
	ds_load_b32 v6, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_125
.LBB1_746:
	s_and_b64 s[56:57], s[12:13], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_748
; %bb.747:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB1_748:                              ; %Flow2187
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_126
.LBB1_749:
	s_mul_i32 s34, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s34, 0x80
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
	ds_load_b32 v6, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_127
.LBB1_750:
	s_and_b64 s[56:57], s[10:11], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_752
; %bb.751:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB1_752:                              ; %Flow2185
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB1_128
.LBB1_753:
	s_mul_i32 s34, s40, 0x180
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s34, 0x100
	buffer_load_b32 v7, v33, s[44:47], s34 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v33, s[44:47], s34 offen
	ds_load_b32 v6, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[34:35], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_129
.LBB1_754:
	s_and_b64 s[56:57], s[8:9], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[34:35], s[56:57]
	s_cbranch_execz .LBB1_756
; %bb.755:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB1_756:                              ; %Flow2183
	s_or_b64 exec, exec, s[34:35]
	s_cbranch_execz .LBB1_130
	s_branch .LBB1_131
.LBB1_757:
	s_and_b64 s[58:59], s[14:15], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_759
; %bb.758:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB1_759:                              ; %Flow2181
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_133
.LBB1_760:
	s_mul_i32 s56, s40, 0x1c0
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_134
.LBB1_761:
	s_and_b64 s[58:59], s[12:13], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_763
; %bb.762:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:128
.LBB1_763:                              ; %Flow2179
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_135
.LBB1_764:
	s_mul_i32 s56, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x80
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_136
.LBB1_765:
	s_and_b64 s[58:59], s[10:11], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_767
; %bb.766:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:256
.LBB1_767:                              ; %Flow2177
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_137
.LBB1_768:
	s_mul_i32 s56, s40, 0x1c0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_addk_co_i32 s56, 0x100
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_138
.LBB1_769:
	s_and_b64 s[58:59], s[8:9], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_771
; %bb.770:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:384
.LBB1_771:                              ; %Flow2175
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execz .LBB1_139
	s_branch .LBB1_140
.LBB1_772:
	s_and_b64 s[58:59], s[6:7], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_774
; %bb.773:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:64
.LBB1_774:                              ; %Flow2173
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_142
.LBB1_775:
	s_or_b32 s56, s41, 64
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_143
.LBB1_776:
	s_and_b64 s[58:59], s[4:5], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_778
; %bb.777:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:192
.LBB1_778:                              ; %Flow2171
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_144
.LBB1_779:
	s_or_b32 s56, s41, 0xc0
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_145
.LBB1_780:
	s_and_b64 s[58:59], s[2:3], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[56:57], s[58:59]
	s_cbranch_execz .LBB1_782
; %bb.781:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:320
.LBB1_782:                              ; %Flow2169
	s_or_b64 exec, exec, s[56:57]
	s_cbranch_execnz .LBB1_146
.LBB1_783:
	s_add_co_i32 s56, s41, 0x140
	buffer_load_b32 v9, v33, s[44:47], s56 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v33, s[44:47], s56 offen
	ds_load_b32 v8, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[56:57], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_147
.LBB1_784:
	s_and_b64 s[56:57], s[0:1], s[26:27]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_786
; %bb.785:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	global_load_b32 v9, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v8, v9
	global_store_b32 v[0:1], v9, off offset:448
.LBB1_786:                              ; %Flow2167
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_148
	s_branch .LBB1_149
.LBB1_787:
	s_and_b64 s[56:57], s[6:7], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_789
; %bb.788:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:64
.LBB1_789:                              ; %Flow2165
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_151
.LBB1_790:
	s_mul_i32 s26, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s26, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_152
.LBB1_791:
	s_and_b64 s[56:57], s[4:5], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_793
; %bb.792:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:192
.LBB1_793:                              ; %Flow2163
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_153
.LBB1_794:
	s_mul_i32 s26, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_154
.LBB1_795:
	s_and_b64 s[56:57], s[2:3], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[56:57]
	s_cbranch_execz .LBB1_797
; %bb.796:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:320
.LBB1_797:                              ; %Flow2161
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_155
.LBB1_798:
	s_mul_i32 s26, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_156
.LBB1_799:
	s_and_b64 s[28:29], s[0:1], s[28:29]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_801
; %bb.800:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_801:                              ; %Flow2159
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_157
	s_branch .LBB1_158
.LBB1_802:
	s_and_b64 s[28:29], s[6:7], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_804
; %bb.803:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB1_804:                              ; %Flow2157
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_160
.LBB1_805:
	s_mul_i32 s26, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s26, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_161
.LBB1_806:
	s_and_b64 s[28:29], s[4:5], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_808
; %bb.807:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB1_808:                              ; %Flow2155
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_162
.LBB1_809:
	s_mul_i32 s26, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_163
.LBB1_810:
	s_and_b64 s[28:29], s[2:3], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_812
; %bb.811:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB1_812:                              ; %Flow2153
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_164
.LBB1_813:
	s_mul_i32 s26, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_165
.LBB1_814:
	s_and_b64 s[28:29], s[0:1], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_816
; %bb.815:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_816:                              ; %Flow2151
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_166
	s_branch .LBB1_167
.LBB1_817:
	s_and_b64 s[28:29], s[6:7], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_819
; %bb.818:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB1_819:                              ; %Flow2149
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_169
.LBB1_820:
	s_mul_i32 s26, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s26, 64
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:1280
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_170
.LBB1_821:
	s_and_b64 s[28:29], s[4:5], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_823
; %bb.822:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB1_823:                              ; %Flow2147
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_171
.LBB1_824:
	s_mul_i32 s26, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0xc0
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:2560
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_172
.LBB1_825:
	s_and_b64 s[28:29], s[2:3], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_827
; %bb.826:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB1_827:                              ; %Flow2145
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB1_173
.LBB1_828:
	s_mul_i32 s26, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s26, 0x140
	buffer_load_b32 v1, v33, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v33, s[44:47], s26 offen
	ds_load_b32 v0, v86 offset:3840
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[26:27], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_174
.LBB1_829:
	s_and_b64 s[28:29], s[0:1], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB1_831
; %bb.830:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_831:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[26:27]
	s_cbranch_execz .LBB1_175
	s_branch .LBB1_176
.Lfunc_end1:
	.size	iu4_w64_r2, .Lfunc_end1-iu4_w64_r2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r2
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 141
		.amdhsa_next_free_sgpr 73
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-iu4_w64_r2)<<4)&4080)>>4
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
	.set .Liu4_w64_r2.num_vgpr, 141
	.set .Liu4_w64_r2.num_agpr, 0
	.set .Liu4_w64_r2.numbered_sgpr, 73
	.set .Liu4_w64_r2.num_named_barrier, 0
	.set .Liu4_w64_r2.private_seg_size, 0
	.set .Liu4_w64_r2.uses_vcc, 1
	.set .Liu4_w64_r2.uses_flat_scratch, 0
	.set .Liu4_w64_r2.has_dyn_sized_stack, 0
	.set .Liu4_w64_r2.has_recursion, 0
	.set .Liu4_w64_r2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 36308
; TotalNumSgprs: 75
; NumVgprs: 141
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 35
; NumSGPRsForWavesPerEU: 75
; NumVGPRsForWavesPerEU: 141
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	iu4_w64_r2_add          ; -- Begin function iu4_w64_r2_add
	.globl	iu4_w64_r2_add
	.p2align	8
	.type	iu4_w64_r2_add,@function
iu4_w64_r2_add:                         ; @iu4_w64_r2_add
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[40:42], s[0:1], 0x18
	s_lshl_b32 s8, ttmp9, 7
	s_lshl_b32 s10, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s8, s40
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_ge_i32 s10, s42
	s_cselect_b64 s[4:5], -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_3
; %bb.1:                                ; %Flow1141
	s_and_not1_b64 vcc, exec, s[2:3]
	s_cbranch_vccz .LBB2_4
.LBB2_2:                                ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB2_3:                                ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb1ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.1.critedge.i
	s_cbranch_execnz .LBB2_2
.LBB2_4:                                ; %.preheader514.i.i
	s_clause 0x1
	s_load_b128 s[48:51], s[0:1], 0x0
	s_load_b64 s[52:53], s[0:1], 0x10
	v_lshrrev_b32_e32 v1, 1, v0
	s_ashr_i32 s0, s41, 31
	v_and_b32_e32 v5, 1, v0
	s_lshr_b32 s0, s0, 24
	s_add_co_i32 s4, s40, -1
	v_or_b32_e32 v6, s10, v1
	v_or_b32_e32 v7, s8, v1
	s_add_co_i32 s0, s41, s0
	v_lshlrev_b32_e32 v84, 2, v5
	s_ashr_i32 s33, s0, 8
	v_lshlrev_b32_e32 v8, 3, v1
	v_min_i32_e32 v9, s4, v7
	s_mul_i32 s6, s33, 0x88
	v_cmp_gt_u32_e64 s[24:25], 0x100, v0
	v_mov_b32_e32 v2, 0
	v_cmp_gt_i32_e64 s[28:29], s42, v6
	v_add3_u32 v85, 0, v8, v84
	v_cmp_gt_i32_e64 s[26:27], s40, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[3:4], null, 0x48, v6, s[50:51]
	v_mad_co_i64_i32 v[42:43], null, s6, v9, s[48:49]
	v_lshlrev_b32_e32 v86, 4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v44, vcc, v3, v84
	v_add_co_ci_u32_e64 v45, null, 0, v4, vcc
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB2_8
; %bb.5:                                ; %.lr.ph.i.i
	s_and_saveexec_b64 s[2:3], s[28:29]
	s_cbranch_execz .LBB2_7
; %bb.6:
	global_load_b32 v2, v[44:45], off
.LBB2_7:                                ; %.preheader511.loopexit.i.i
	s_or_b64 exec, exec, s[2:3]
	global_load_b32 v3, v[42:43], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v3, v86, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v3, v3.l
	v_cndmask_b32_e64 v3, 0, v3, s[26:27]
	ds_store_2addr_stride64_b32 v85, v2, v3 offset0:48 offset1:56
.LBB2_8:                                ; %Flow1140
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v10, 2, v0
	v_and_b32_e32 v2, 3, v0
	s_add_co_i32 s2, s42, -1
	v_and_b32_e32 v13, 0x60, v1
	v_add_nc_u32_e32 v18, s10, v1
	v_add_nc_u32_e32 v11, s10, v10
	v_add_nc_u32_e32 v3, s8, v10
	v_lshlrev_b32_e32 v2, 3, v2
	v_add_nc_u32_e32 v1, s8, v1
	v_lshrrev_b32_e32 v17, 5, v0
	v_add_nc_u32_e32 v12, 64, v11
	v_add_nc_u32_e32 v4, 64, v3
	v_min_i32_e32 v5, s2, v11
	v_min_i32_e32 v3, s4, v3
	v_bfe_u32 v14, v0, 1, 1
	v_min_i32_e32 v6, s2, v12
	v_min_i32_e32 v4, s4, v4
	v_lshlrev_b32_e32 v15, 7, v0
	v_mad_co_u64_u32 v[36:37], null, 0x48, v5, v[2:3]
	v_mad_co_u64_u32 v[37:38], null, s6, v3, v[2:3]
	v_mad_co_u64_u32 v[38:39], null, 0x48, v6, v[2:3]
	v_mad_co_u64_u32 v[39:40], null, s6, v4, v[2:3]
	v_and_b32_e32 v16, 60, v0
	v_and_b32_e32 v123, 12, v10
	v_min_i32_e32 v10, s4, v1
	v_or_b32_e32 v19, 8, v17
	global_load_b64 v[2:3], v36, s[50:51] offset:8
	global_load_b64 v[4:5], v37, s[48:49] offset:8
	global_load_b64 v[6:7], v38, s[50:51] offset:8
	global_load_b64 v[8:9], v39, s[48:49] offset:8
	v_and_or_b32 v15, 0x80, v15, v16
	v_and_or_b32 v16, v17, 6, v14
	v_mad_co_u64_u32 v[33:34], null, s6, v10, s[48:49]
	v_min_i32_e32 v88, s2, v18
	v_cmp_gt_i32_e64 s[2:3], s40, v1
	v_and_or_b32 v1, v19, 14, v14
	v_ashrrev_i32_e32 v10, 31, v10
	v_lshl_or_b32 v14, v16, 8, v15
	v_cmp_gt_i32_e64 s[4:5], s42, v11
	v_lshlrev_b32_e32 v94, 2, v0
	v_lshl_or_b32 v1, v1, 8, v15
	v_mad_co_u64_u32 v[40:41], null, s6, v10, v[34:35]
	v_cmp_gt_i32_e64 s[6:7], s42, v12
	v_add_nc_u32_e32 v91, 0, v14
	v_add_co_u32 v50, s[14:15], s50, v36
	v_add_nc_u32_e32 v93, 0, v1
	v_add_co_ci_u32_e64 v51, null, s51, 0, s[14:15]
	v_add_co_u32 v52, s[14:15], s48, v37
	v_and_b32_e32 v124, 15, v0
	v_and_b32_e32 v87, 0xfc, v94
	v_or_b32_e32 v16, v123, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v53, null, s49, 0, s[14:15]
	v_add_co_u32 v46, s[14:15], s50, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v47, null, s51, 0, s[14:15]
	v_add_co_u32 v48, s[14:15], s48, v39
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v98, 0
	s_cmp_gt_i32 s41, 0xff
	v_mov_b32_e32 v95, 0
	v_cmp_gt_i32_e64 s[0:1], s42, v18
	v_lshl_add_u32 v89, v124, 3, 0
	v_lshl_or_b32 v90, v13, 5, v87
	v_lshl_add_u32 v92, v16, 3, 0
	v_add_co_ci_u32_e64 v49, null, s49, 0, s[14:15]
	s_cselect_b64 s[54:55], -1, 0
	s_ashr_i32 s43, s42, 31
	s_mov_b32 s13, 0
	s_cmp_lt_i32 s41, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s[4:5]
	v_cndmask_b32_e64 v3, 0, v3, s[4:5]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v1, 0, v6, s[6:7]
	v_cndmask_b32_e64 v6, 0, v7, s[6:7]
	ds_store_b32 v91, v4 offset:4096
	ds_store_2addr_b32 v91, v2, v3 offset1:16
	ds_store_b32 v91, v5 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v93, v8 offset:4096
	ds_store_2addr_b32 v93, v1, v6 offset1:16
	ds_store_b32 v93, v9 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_17
; %bb.9:                                ; %.preheader510.lr.ph.i.i
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	s_movk_i32 s30, 0x1000
	s_movk_i32 s9, 0x3000
	s_movk_i32 s11, 0x3800
	s_mov_b32 s31, 0
	s_mov_b32 s14, s13
	s_branch .LBB2_11
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_and_b64 vcc, exec, s[16:17]
	s_mov_b32 s14, s34
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_17
.LBB2_11:                               ; %.preheader510.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_13 Depth 2
	s_mov_b32 s15, s13
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s34, s14, 1
	s_mul_u64 s[18:19], s[14:15], 0x88
	s_lshl_b32 s15, s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s34, s33
	s_mov_b32 s35, 0
	s_cselect_b64 s[16:17], -1, 0
	s_add_nc_u64 s[18:19], s[48:49], s[18:19]
	s_mov_b64 s[22:23], 0
	s_mov_b64 s[20:21], -1
	s_branch .LBB2_13
.LBB2_12:                               ;   in Loop: Header=BB2_13 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v68, v82, v66
	v_mul_f32_e32 v69, v80, v66
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v67, v67
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v70, v78, v66
	v_cvt_f32_i32_e32 v31, v31
	v_mul_f32_e32 v29, v68, v29
	v_mul_f32_e32 v30, v69, v30
	v_mul_f32_e32 v68, v83, v66
	v_mul_f32_e32 v69, v81, v66
	v_cvt_f32_i32_e32 v32, v32
	v_mul_f32_e32 v31, v70, v31
	v_cvt_f32_i32_e32 v25, v25
	v_fmac_f32_e32 v29, v68, v67
	v_fmac_f32_e32 v30, v69, v67
	v_mul_f32_e32 v68, v76, v66
	v_mul_f32_e32 v69, v79, v66
	v_cvt_f32_i32_e32 v26, v26
	v_add_f32_e32 v128, v128, v29
	v_add_f32_e32 v127, v127, v30
	v_mul_f32_e32 v29, v68, v32
	v_mul_f32_e32 v30, v77, v66
	v_fmac_f32_e32 v31, v69, v67
	s_wait_dscnt 0x6
	v_mul_f32_e32 v32, v82, v64
	v_mul_f32_e32 v68, v80, v64
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v29, v30, v67
	v_add_f32_e32 v126, v126, v31
	v_cvt_f32_i32_e32 v30, v65
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v26, v68, v26
	v_mul_f32_e32 v31, v83, v64
	v_mul_f32_e32 v32, v81, v64
	v_add_f32_e32 v125, v125, v29
	v_mul_f32_e32 v29, v78, v64
	v_cvt_f32_i32_e32 v28, v28
	v_fmac_f32_e32 v25, v31, v30
	v_fmac_f32_e32 v26, v32, v30
	v_mul_f32_e32 v31, v76, v64
	v_mul_f32_e32 v27, v29, v27
	v_mul_f32_e32 v29, v79, v64
	v_add_f32_e32 v121, v121, v25
	v_add_f32_e32 v122, v122, v26
	v_mul_f32_e32 v25, v31, v28
	v_mul_f32_e32 v26, v77, v64
	v_fmac_f32_e32 v27, v29, v30
	s_wait_dscnt 0x5
	v_mul_f32_e32 v28, v82, v62
	v_mul_f32_e32 v29, v80, v62
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v25, v26, v30
	v_add_f32_e32 v119, v119, v27
	v_cvt_f32_i32_e32 v26, v63
	v_mul_f32_e32 v21, v28, v21
	v_mul_f32_e32 v22, v29, v22
	v_mul_f32_e32 v27, v78, v62
	v_cvt_f32_i32_e32 v23, v23
	v_mul_f32_e32 v28, v83, v62
	v_mul_f32_e32 v29, v81, v62
	v_mul_f32_e32 v31, v76, v62
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v23, v27, v23
	v_mul_f32_e32 v27, v79, v62
	v_fmac_f32_e32 v21, v28, v26
	v_fmac_f32_e32 v22, v29, v26
	v_mul_f32_e32 v24, v31, v24
	v_mul_f32_e32 v28, v77, v62
	v_fmac_f32_e32 v23, v27, v26
	v_add_f32_e32 v117, v117, v21
	v_add_f32_e32 v118, v118, v22
	s_wait_dscnt 0x4
	v_mul_f32_e32 v21, v82, v34
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v80, v34
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v24, v28, v26
	v_add_f32_e32 v116, v116, v23
	v_cvt_f32_i32_e32 v23, v35
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v83, v34
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v78, v34
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v115, v115, v24
	v_mul_f32_e32 v24, v81, v34
	v_fmac_f32_e32 v17, v21, v23
	v_mul_f32_e32 v21, v76, v34
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v79, v34
	v_fmac_f32_e32 v18, v24, v23
	v_add_f32_e32 v113, v113, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v77, v34
	v_fmac_f32_e32 v19, v22, v23
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v66, v60
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v66, v58
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v114, v114, v18
	v_fmac_f32_e32 v17, v20, v23
	v_add_f32_e32 v111, v111, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v66, v61
	v_mul_f32_e32 v19, v66, v59
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v66, v54
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v112, v112, v17
	v_fmac_f32_e32 v13, v18, v67
	v_fmac_f32_e32 v14, v19, v67
	v_mul_f32_e32 v17, v66, v55
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v64, v60
	v_mul_f32_e32 v20, v64, v58
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v109, v109, v13
	v_fmac_f32_e32 v15, v17, v67
	v_mul_f32_e32 v13, v64, v61
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v64, v59
	v_add_f32_e32 v110, v110, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v30
	v_mul_f32_e32 v13, v64, v54
	v_fmac_f32_e32 v10, v17, v30
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v64, v56
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v105, v105, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v106, v106, v10
	v_mul_f32_e32 v10, v64, v55
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v62, v60
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v30
	v_mul_f32_e32 v10, v62, v58
	v_mul_f32_e32 v13, v64, v57
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v62, v61
	v_add_f32_e32 v104, v104, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v62, v59
	v_mul_f32_e32 v10, v62, v54
	v_fmac_f32_e32 v5, v12, v26
	v_mul_f32_e32 v12, v62, v56
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v30
	v_fmac_f32_e32 v6, v9, v26
	v_add_f32_e32 v101, v101, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v62, v55
	v_mul_f32_e32 v9, v62, v57
	v_mul_f32_e32 v21, v66, v56
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v103, v103, v11
	v_mul_f32_e32 v10, v34, v60
	v_mul_f32_e32 v11, v34, v58
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v26
	v_fmac_f32_e32 v7, v9, v26
	v_mul_f32_e32 v8, v34, v54
	v_mul_f32_e32 v9, v34, v56
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v66, v57
	v_add_f32_e32 v102, v102, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v34, v61
	v_mul_f32_e32 v10, v34, v59
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v34, v55
	v_mul_f32_e32 v9, v34, v57
	v_fmac_f32_e32 v16, v18, v67
	v_fmac_f32_e32 v1, v6, v23
	v_fmac_f32_e32 v2, v10, v23
	v_fmac_f32_e32 v3, v8, v23
	v_fmac_f32_e32 v4, v9, v23
	v_add_f32_e32 v120, v120, v25
	v_add_f32_e32 v107, v107, v15
	v_add_f32_e32 v108, v108, v16
	v_add_f32_e32 v99, v99, v5
	v_add_f32_e32 v100, v100, v7
	v_add_f32_e32 v95, v95, v1
	v_add_f32_e32 v98, v98, v2
	v_add_f32_e32 v96, v96, v3
	v_add_f32_e32 v97, v97, v4
	s_xor_b64 s[22:23], s[20:21], -1
	s_mov_b32 s35, 1
	s_mov_b64 s[20:21], 0
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[22:23], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_10
.LBB2_13:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s12, s35, s15
	s_lshl_b32 s36, s35, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[12:13], s[42:43]
	s_mov_b32 s37, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[38:39], 0x48
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[50:51], s[38:39]
	v_add_nc_u32_e32 v56, s30, v90
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[44:45], s38, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s39, 0, s[44:45]
	v_add_co_u32 v3, s[44:45], s36, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s37, 0, s[44:45]
	v_add_co_u32 v5, s[44:45], s38, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s39, 0, s[44:45]
	v_add_co_u32 v7, s[38:39], s36, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s37, 0, s[38:39]
	global_load_b64 v[74:75], v[1:2], off offset:40
	global_load_b64 v[68:69], v[3:4], off offset:40
	global_load_b64 v[72:73], v[5:6], off offset:40
	global_load_b64 v[70:71], v[7:8], off offset:40
	v_add_nc_u32_e32 v57, s31, v87
	ds_load_2addr_stride64_b32 v[34:35], v56 offset1:2
	ds_load_2addr_stride64_b32 v[1:2], v57 offset1:2
	ds_load_2addr_stride64_b32 v[54:55], v57 offset0:4 offset1:6
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v2, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v54, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v55, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v2, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v54, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v55, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[34:35], v56 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[54:55], v57 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[56:57], v57 offset0:5 offset1:7
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v54, 0x2000, v91
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v74, s[4:5]
	v_add_nc_u32_e32 v35, 0x4000, v91
	v_cndmask_b32_e64 v55, 0, v75, s[4:5]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v54, v68, v69 offset1:16
	ds_store_2addr_b32 v35, v34, v55 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v72, s[6:7]
	v_cndmask_b32_e64 v35, 0, v73, s[6:7]
	v_add_nc_u32_e32 v54, 0x4000, v93
	v_add_nc_u32_e32 v55, 0x2000, v93
	ds_store_2addr_b32 v54, v34, v35 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v55, v70, v71 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[30:31], s[22:23], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[30:31]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_15
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=2
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[12:13], s[42:43]
	s_add_co_i32 s12, s35, s14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_mul_u64 s[38:39], s[12:13], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[50:51], s[36:37]
	s_xor_b32 s35, s35, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[56:57], null, 0x48, v88, s[36:37]
	v_add_co_u32 v34, s[44:45], s36, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, s37, 0, s[44:45]
	s_add_nc_u64 s[38:39], s[48:49], s[38:39]
	s_lshl_b32 s44, s35, 6
	s_mov_b32 s45, s13
	v_add_co_u32 v56, vcc, v56, v84
	s_mulk_i32 s12, 0x88
	v_add_co_u32 v54, s[46:47], s36, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[38:39], s[44:45]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, 0, v57, vcc
	v_add_co_u32 v41, vcc, v33, s12
	v_add_co_ci_u32_e64 v55, null, s37, 0, s[46:47]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v58, s[36:37], s38, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v59, null, s39, 0, s[36:37]
	v_add_co_u32 v60, s[36:37], s38, v39
	s_lshl_b32 s12, s35, 2
	v_add_co_ci_u32_e64 v61, null, s39, 0, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v62, vcc, v41, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, 0, v63, vcc
	s_clause 0x1
	global_load_b64 v[74:75], v[34:35], off offset:8
	global_load_b64 v[72:73], v[54:55], off offset:8
	s_clause 0x1
	global_load_b64 v[68:69], v[58:59], off offset:8
	global_load_b64 v[70:71], v[60:61], off offset:8
	global_load_b32 v41, v[56:57], off
	global_load_b32 v129, v[62:63], off
.LBB2_15:                               ; %.preheader508.i.i
                                        ;   in Loop: Header=BB2_13 Depth=2
	v_add_nc_u32_e32 v58, 0, v90
	v_add_nc_u32_e32 v59, 0, v87
	s_xor_b64 s[30:31], s[30:31], -1
	s_and_b64 s[36:37], s[20:21], exec
	s_cselect_b32 s12, s9, 0x3400
	ds_load_2addr_stride64_b32 v[34:35], v58 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[54:55], v59 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:68 offset1:70
	s_cselect_b32 s35, s11, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[34:35], v58 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[54:55], v59 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:69 offset1:71
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v56, s35, v92
	v_add_nc_u32_e32 v34, s12, v89
	s_and_not1_b64 vcc, exec, s[30:31]
	s_movk_i32 s30, 0x2000
	s_movk_i32 s31, 0x4000
	ds_load_2addr_b32 v[82:83], v56 offset1:1
	ds_load_2addr_b32 v[80:81], v56 offset0:2 offset1:3
	ds_load_2addr_b32 v[78:79], v56 offset0:4 offset1:5
	ds_load_2addr_b32 v[76:77], v56 offset0:6 offset1:7
	ds_load_2addr_b32 v[66:67], v34 offset1:1
	ds_load_2addr_b32 v[64:65], v34 offset0:32 offset1:33
	ds_load_2addr_b32 v[62:63], v34 offset0:64 offset1:65
	ds_load_2addr_b32 v[34:35], v34 offset0:96 offset1:97
	ds_load_2addr_b32 v[60:61], v56 offset0:32 offset1:33
	ds_load_2addr_b32 v[58:59], v56 offset0:34 offset1:35
	ds_load_2addr_b32 v[54:55], v56 offset0:36 offset1:37
	ds_load_2addr_b32 v[56:57], v56 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.16:                               ; %.preheader509.i.i
                                        ;   in Loop: Header=BB2_13 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v130, v86, v129
	s_and_b64 s[22:23], s[22:23], exec
	v_cndmask_b32_e64 v74, 0, v74, s[4:5]
	v_cndmask_b32_e64 v75, 0, v75, s[4:5]
	s_cselect_b32 s12, s11, 0x3c00
	v_cvt_f32_f16_e64 v130, v130.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v131, s12, v85
	ds_store_b32 v91, v68 offset:4096
	ds_store_2addr_b32 v91, v74, v75 offset1:16
	v_cndmask_b32_e64 v68, 0, v72, s[6:7]
	v_cndmask_b32_e64 v72, 0, v73, s[6:7]
	v_cndmask_b32_e64 v130, 0, v130, s[2:3]
	s_cselect_b32 s12, s9, 0x3400
	s_mov_b32 s31, 0
	s_movk_i32 s30, 0x1000
	v_cndmask_b32_e64 v73, 0, v41, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s12, v85
	ds_store_b32 v91, v69 offset:4160
	ds_store_b32 v93, v70 offset:4096
	ds_store_2addr_b32 v93, v68, v72 offset1:16
	ds_store_b32 v93, v71 offset:4160
	ds_store_b32 v74, v73
	ds_store_b32 v131, v130
	s_branch .LBB2_12
.LBB2_17:                               ; %._crit_edge580.i.i
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v2, 2, v124
	v_lshrrev_b32_e32 v0, 4, v0
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s41, s40, 31
	v_mul_u32_u24_e32 v1, 0x500, v1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[12:13], s[40:41], s[10:11]
	v_lshlrev_b32_e32 v4, 2, v0
	v_or_b32_e32 v76, s10, v0
	v_mul_lo_u32 v3, s40, v0
	v_add3_u32 v2, 0, v1, v2
	s_ashr_i32 s9, s8, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[12:13], s[12:13], 2
	s_lshl_b64 s[14:15], s[8:9], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[52:53], s[12:13]
	v_mad_u32_u24 v1, 0x50, v123, v2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[44:45], s[12:13], s[14:15]
	s_add_co_i32 s11, s8, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s45, s45, 0xffff
	s_cmp_gt_i32 s11, s40
	ds_store_2addr_b32 v1, v128, v127 offset1:20
	ds_store_2addr_b32 v1, v126, v125 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v124
	s_cselect_b64 s[12:13], -1, 0
	s_add_co_i32 s9, s10, 0x80
	v_or_b32_e32 v34, s8, v124
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s9, s42
	v_add3_u32 v32, 0, v1, v4
	v_mad_co_i64_i32 v[0:1], null, s40, v76, 0
	s_cselect_b64 s[8:9], -1, 0
	v_ashrrev_i32_e32 v35, 31, v34
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[12:13], s[12:13], s[8:9]
	v_cmp_gt_i32_e64 s[8:9], s40, v34
	v_cmp_gt_i32_e64 s[30:31], s42, v76
	s_mov_b32 s47, 0x31004000
	s_mov_b32 s46, -1
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_mov_b64 s[10:11], -1
	v_add_co_u32 v0, vcc, s52, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s53, v1, vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_21
; %bb.18:
	s_and_b64 s[14:15], s[8:9], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[10:11], s[14:15]
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v1, v6, vcc
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off
.LBB2_20:                               ; %Flow1134
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[10:11]
	s_mov_b64 s[10:11], 0
.LBB2_21:                               ; %Flow1135
	v_add_lshl_u32 v73, v3, v124, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_23
; %bb.22:
	buffer_load_b32 v3, v73, s[44:47], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	buffer_store_b32 v3, v73, s[44:47], null offen
.LBB2_23:
	ds_load_b32 v3, v32 offset:1280
	s_wait_dscnt 0x1
	v_or_b32_e32 v4, 32, v34
	v_cndmask_b32_e64 v74, 0, 1, s[12:13]
	s_and_not1_b64 vcc, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[10:11], s40, v4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_27
; %bb.24:
	s_and_b64 s[14:15], s[10:11], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB2_26
; %bb.25:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	global_load_b32 v6, v[4:5], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v3, v6
	global_store_b32 v[4:5], v6, off offset:128
.LBB2_26:                               ; %Flow1132
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
.LBB2_27:                               ; %Flow1133
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_29
; %bb.28:
	s_movk_i32 s12, 0x80
	buffer_load_b32 v4, v73, s[44:47], s12 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v3, v4
	buffer_store_b32 v3, v73, s[44:47], s12 offen
.LBB2_29:
	s_wait_dscnt 0x0
	ds_load_b32 v3, v32 offset:2560
	v_or_b32_e32 v4, 64, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[14:15], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[12:13], s40, v4
	s_cbranch_vccnz .LBB2_33
; %bb.30:
	s_and_b64 s[16:17], s[12:13], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB2_32
; %bb.31:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	global_load_b32 v6, v[4:5], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v3, v6
	global_store_b32 v[4:5], v6, off offset:256
.LBB2_32:                               ; %Flow1130
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[14:15]
	s_mov_b64 s[14:15], 0
.LBB2_33:                               ; %Flow1131
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_35
; %bb.34:
	s_movk_i32 s14, 0x100
	buffer_load_b32 v4, v73, s[44:47], s14 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v3, v4
	buffer_store_b32 v3, v73, s[44:47], s14 offen
.LBB2_35:
	s_wait_dscnt 0x0
	ds_load_b32 v3, v32 offset:3840
	v_or_b32_e32 v4, 0x60, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[14:15], s40, v4
	s_cbranch_vccnz .LBB2_39
; %bb.36:
	s_and_b64 s[18:19], s[14:15], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_38
; %bb.37:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	global_load_b32 v6, v[4:5], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v3, v6
	global_store_b32 v[4:5], v6, off offset:384
.LBB2_38:                               ; %Flow1128
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_39:                               ; %Flow1129
	v_mul_u32_u24_e32 v4, 0x50, v123
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_41
; %bb.40:
	s_movk_i32 s16, 0x180
	buffer_load_b32 v5, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v3, v5
	buffer_store_b32 v3, v73, s[44:47], s16 offen
.LBB2_41:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v75, v2, v4
	v_or_b32_e32 v5, 16, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[34:35], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v121, v122 offset1:20
	ds_store_2addr_b32 v75, v119, v120 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[16:17], s52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s53, v3, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_45
; %bb.42:
	s_and_b64 s[18:19], s[8:9], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_44
; %bb.43:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off
.LBB2_44:                               ; %Flow1126
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_45:                               ; %Flow1127
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_lshl_b32 s41, s40, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_47
; %bb.46:
	buffer_load_b32 v5, v73, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s41 offen
.LBB2_47:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_51
; %bb.48:
	s_and_b64 s[18:19], s[10:11], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_50
; %bb.49:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:128
.LBB2_50:                               ; %Flow1124
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_51:                               ; %Flow1125
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_53
; %bb.52:
	s_add_co_i32 s16, s41, 0x80
	buffer_load_b32 v5, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB2_53:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_57
; %bb.54:
	s_and_b64 s[18:19], s[12:13], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_56
; %bb.55:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:256
.LBB2_56:                               ; %Flow1122
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_57:                               ; %Flow1123
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_59
; %bb.58:
	s_add_co_i32 s16, s41, 0x100
	buffer_load_b32 v5, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB2_59:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_63
; %bb.60:
	s_and_b64 s[18:19], s[14:15], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_62
; %bb.61:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:384
.LBB2_62:                               ; %Flow1120
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_63:                               ; %Flow1121
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_65
; %bb.64:
	s_add_co_i32 s16, s41, 0x180
	buffer_load_b32 v5, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB2_65:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 32, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[36:37], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v117, v118 offset1:20
	ds_store_2addr_b32 v75, v116, v115 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[16:17], s52, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s53, v5, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_69
; %bb.66:
	s_and_b64 s[18:19], s[8:9], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_68
; %bb.67:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB2_68:                               ; %Flow1118
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_69:                               ; %Flow1119
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_lshl_b32 s58, s40, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_71
; %bb.70:
	buffer_load_b32 v7, v73, s[44:47], s58 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s58 offen
.LBB2_71:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_75
; %bb.72:
	s_and_b64 s[18:19], s[10:11], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_74
; %bb.73:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB2_74:                               ; %Flow1116
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_75:                               ; %Flow1117
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_77
; %bb.76:
	s_add_co_i32 s16, s58, 0x80
	buffer_load_b32 v7, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB2_77:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_81
; %bb.78:
	s_and_b64 s[18:19], s[12:13], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_80
; %bb.79:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB2_80:                               ; %Flow1114
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_81:                               ; %Flow1115
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_83
; %bb.82:
	s_add_co_i32 s16, s58, 0x100
	buffer_load_b32 v7, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB2_83:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_87
; %bb.84:
	s_and_b64 s[18:19], s[14:15], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_86
; %bb.85:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB2_86:                               ; %Flow1112
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_87:                               ; %Flow1113
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_89
; %bb.88:
	s_add_co_i32 s16, s58, 0x180
	buffer_load_b32 v7, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB2_89:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 48, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[38:39], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v113, v114 offset1:20
	ds_store_2addr_b32 v75, v111, v112 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[16:17], s52, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s53, v7, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_93
; %bb.90:
	s_and_b64 s[18:19], s[8:9], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_92
; %bb.91:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB2_92:                               ; %Flow1110
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_93:                               ; %Flow1111
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_mul_i32 s59, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_95
; %bb.94:
	buffer_load_b32 v9, v73, s[44:47], s59 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s59 offen
.LBB2_95:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_99
; %bb.96:
	s_and_b64 s[18:19], s[10:11], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_98
; %bb.97:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:128
.LBB2_98:                               ; %Flow1108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_99:                               ; %Flow1109
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_101
; %bb.100:
	s_add_co_i32 s16, s59, 0x80
	buffer_load_b32 v9, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB2_101:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_105
; %bb.102:
	s_and_b64 s[18:19], s[12:13], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_104
; %bb.103:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:256
.LBB2_104:                              ; %Flow1106
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_105:                              ; %Flow1107
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_107
; %bb.106:
	s_add_co_i32 s16, s59, 0x100
	buffer_load_b32 v9, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB2_107:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB2_111
; %bb.108:
	s_and_b64 s[18:19], s[14:15], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB2_110
; %bb.109:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:384
.LBB2_110:                              ; %Flow1104
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB2_111:                              ; %Flow1105
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_113
; %bb.112:
	s_add_co_i32 s16, s59, 0x180
	buffer_load_b32 v9, v73, s[44:47], s16 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB2_113:                              ; %.preheader505.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 16, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[18:19], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[16:17], s40, v9
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v109, v110 offset1:20
	ds_store_2addr_b32 v75, v107, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_117
; %bb.114:
	s_and_b64 s[20:21], s[16:17], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB2_116
; %bb.115:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:64
.LBB2_116:                              ; %Flow1102
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB2_117:                              ; %Flow1103
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_119
; %bb.118:
	s_mov_b32 s18, 64
	buffer_load_b32 v9, v73, s[44:47], s18 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s18 offen
.LBB2_119:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_or_b32_e32 v9, 48, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[20:21], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[18:19], s40, v9
	s_cbranch_vccnz .LBB2_123
; %bb.120:
	s_and_b64 s[22:23], s[18:19], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB2_122
; %bb.121:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:192
.LBB2_122:                              ; %Flow1100
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB2_123:                              ; %Flow1101
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_125
; %bb.124:
	s_movk_i32 s20, 0xc0
	buffer_load_b32 v9, v73, s[44:47], s20 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s20 offen
.LBB2_125:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_or_b32_e32 v9, 0x50, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[22:23], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[20:21], s40, v9
	s_cbranch_vccnz .LBB2_129
; %bb.126:
	s_and_b64 s[56:57], s[20:21], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[22:23], s[56:57]
	s_cbranch_execz .LBB2_128
; %bb.127:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:320
.LBB2_128:                              ; %Flow1098
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB2_129:                              ; %Flow1099
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_131
; %bb.130:
	s_movk_i32 s22, 0x140
	buffer_load_b32 v9, v73, s[44:47], s22 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s22 offen
.LBB2_131:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_or_b32_e32 v9, 0x70, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[56:57], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[22:23], s40, v9
	s_cbranch_vccnz .LBB2_135
; %bb.132:
	s_and_b64 s[56:57], s[22:23], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB2_134
; %bb.133:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	global_load_b32 v9, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v8, v9
	global_store_b32 v[0:1], v9, off offset:448
.LBB2_134:                              ; %Flow1096
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[56:57], 0
.LBB2_135:                              ; %Flow1097
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_137
; %bb.136:
	s_movk_i32 s30, 0x1c0
	buffer_load_b32 v0, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v8, v0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_137:                              ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v105, v106 offset1:20
	ds_store_2addr_b32 v75, v104, v103 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_141
; %bb.138:
	s_and_b64 s[56:57], s[16:17], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB2_140
; %bb.139:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:64
.LBB2_140:                              ; %Flow1094
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_141:                              ; %Flow1095
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_143
; %bb.142:
	s_add_co_i32 s30, s41, 64
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_143:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_147
; %bb.144:
	s_and_b64 s[56:57], s[18:19], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB2_146
; %bb.145:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:192
.LBB2_146:                              ; %Flow1092
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_147:                              ; %Flow1093
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_149
; %bb.148:
	s_add_co_i32 s30, s41, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_149:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_153
; %bb.150:
	s_and_b64 s[56:57], s[20:21], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB2_152
; %bb.151:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:320
.LBB2_152:                              ; %Flow1090
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_153:                              ; %Flow1091
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_155
; %bb.154:
	s_add_co_i32 s30, s41, 0x140
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_155:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_159
; %bb.156:
	s_and_b64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_158
; %bb.157:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_158:                              ; %Flow1088
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_159:                              ; %Flow1089
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_161
; %bb.160:
	s_addk_co_i32 s41, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s41 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s41 offen
.LBB2_161:                              ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v101, v102 offset1:20
	ds_store_2addr_b32 v75, v99, v100 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_165
; %bb.162:
	s_and_b64 s[34:35], s[16:17], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_164
; %bb.163:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB2_164:                              ; %Flow1086
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_165:                              ; %Flow1087
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_167
; %bb.166:
	s_or_b32 s30, s58, 64
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_167:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_171
; %bb.168:
	s_and_b64 s[34:35], s[18:19], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_170
; %bb.169:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB2_170:                              ; %Flow1084
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_171:                              ; %Flow1085
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_173
; %bb.172:
	s_add_co_i32 s30, s58, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_173:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_177
; %bb.174:
	s_and_b64 s[34:35], s[20:21], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_176
; %bb.175:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB2_176:                              ; %Flow1082
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_177:                              ; %Flow1083
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_179
; %bb.178:
	s_add_co_i32 s30, s58, 0x140
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_179:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_183
; %bb.180:
	s_and_b64 s[34:35], s[22:23], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_182
; %bb.181:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_182:                              ; %Flow1080
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_183:                              ; %Flow1081
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_185
; %bb.184:
	s_addk_co_i32 s58, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s58 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s58 offen
.LBB2_185:                              ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v95, v98 offset1:20
	ds_store_2addr_b32 v75, v96, v97 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_189
; %bb.186:
	s_and_b64 s[34:35], s[16:17], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_188
; %bb.187:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB2_188:                              ; %Flow1078
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_189:                              ; %Flow1079
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_191
; %bb.190:
	s_add_co_i32 s30, s59, 64
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_191:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_195
; %bb.192:
	s_and_b64 s[34:35], s[18:19], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_194
; %bb.193:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB2_194:                              ; %Flow1076
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_195:                              ; %Flow1077
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_197
; %bb.196:
	s_add_co_i32 s30, s59, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_197:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_201
; %bb.198:
	s_and_b64 s[34:35], s[20:21], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_200
; %bb.199:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB2_200:                              ; %Flow1074
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_201:                              ; %Flow1075
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_203
; %bb.202:
	s_add_co_i32 s30, s59, 0x140
	buffer_load_b32 v1, v73, s[44:47], s30 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB2_203:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB2_207
; %bb.204:
	s_and_b64 s[34:35], s[22:23], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB2_206
; %bb.205:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_206:                              ; %Flow1072
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB2_207:                              ; %Flow1073
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_209
; %bb.208:
	s_addk_co_i32 s59, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s59 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s59 offen
.LBB2_209:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb1ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b64 s[30:31], s[24:25]
	s_cbranch_execz .LBB2_213
; %bb.210:                              ; %.lr.ph.i.1.i
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[24:25], s[28:29]
	s_cbranch_execz .LBB2_212
; %bb.211:
	global_load_b32 v0, v[44:45], off
.LBB2_212:                              ; %.preheader511.loopexit.i.1.i
	s_or_b64 exec, exec, s[24:25]
	global_load_b32 v1, v[42:43], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v86, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[26:27]
	ds_store_2addr_stride64_b32 v85, v0, v1 offset0:48 offset1:56
.LBB2_213:                              ; %Flow1071
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	global_load_b64 v[0:1], v[50:51], off offset:8
	global_load_b64 v[2:3], v[52:53], off offset:8
	global_load_b64 v[4:5], v[46:47], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v77, 0
	s_mov_b32 s25, 0
	s_and_not1_b64 vcc, exec, s[54:55]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[4:5]
	v_cndmask_b32_e64 v1, 0, v1, s[4:5]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[6:7]
	v_cndmask_b32_e64 v5, 0, v5, s[6:7]
	ds_store_b32 v91, v2 offset:4096
	ds_store_2addr_b32 v91, v0, v1 offset1:16
	ds_store_b32 v91, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v93, v6 offset:4096
	ds_store_2addr_b32 v93, v4, v5 offset1:16
	ds_store_b32 v93, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_222
; %bb.214:                              ; %.preheader510.lr.ph.i.1.i
	v_or_b32_e32 v94, 0xf00, v94
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	s_movk_i32 s38, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s54, 0x3800
	s_mov_b32 s39, 0
	s_mov_b32 s26, s25
	s_branch .LBB2_216
.LBB2_215:                              ;   in Loop: Header=BB2_216 Depth=1
	s_and_not1_b64 vcc, exec, s[28:29]
	s_mov_b32 s26, s55
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_222
.LBB2_216:                              ; %.preheader510.i.1.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_218 Depth 2
	s_mov_b32 s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s26, 1
	s_mul_u64 s[30:31], s[26:27], 0x88
	s_lshl_b32 s27, s26, 1
	s_cmp_eq_u32 s55, s33
	s_mov_b32 s56, 0
	s_cselect_b64 s[28:29], -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[48:49], s[30:31]
	s_mov_b64 s[36:37], 0
	s_mov_b64 s[34:35], -1
	s_branch .LBB2_218
.LBB2_217:                              ;   in Loop: Header=BB2_218 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v57, v71, v55
	v_cvt_f32_i32_e32 v28, v28
	v_mul_f32_e32 v58, v69, v55
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v56, v56
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v59, v70, v55
	v_mul_f32_e32 v28, v57, v28
	v_mul_f32_e32 v57, v72, v55
	v_mul_f32_e32 v29, v58, v29
	v_mul_f32_e32 v58, v65, v55
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v28, v57, v56
	v_fmac_f32_e32 v29, v59, v56
	v_mul_f32_e32 v57, v67, v55
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v119, v119, v28
	v_add_f32_e32 v118, v118, v29
	v_mul_f32_e32 v28, v58, v30
	v_mul_f32_e32 v29, v66, v55
	v_mul_f32_e32 v30, v57, v31
	s_wait_dscnt 0x6
	v_mul_f32_e32 v57, v71, v53
	v_mul_f32_e32 v31, v68, v55
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v56
	v_cvt_f32_i32_e32 v29, v54
	v_mul_f32_e32 v54, v69, v53
	v_mul_f32_e32 v24, v57, v24
	v_mul_f32_e32 v57, v72, v53
	v_fmac_f32_e32 v30, v31, v56
	v_add_f32_e32 v117, v117, v28
	v_mul_f32_e32 v25, v54, v25
	v_mul_f32_e32 v28, v70, v53
	v_fmac_f32_e32 v24, v57, v29
	v_mul_f32_e32 v31, v65, v53
	v_mul_f32_e32 v54, v67, v53
	v_cvt_f32_i32_e32 v27, v27
	v_add_f32_e32 v116, v116, v30
	v_fmac_f32_e32 v25, v28, v29
	v_add_f32_e32 v115, v115, v24
	v_mul_f32_e32 v24, v31, v26
	v_mul_f32_e32 v26, v54, v27
	v_mul_f32_e32 v27, v66, v53
	v_mul_f32_e32 v28, v68, v53
	s_wait_dscnt 0x5
	v_mul_f32_e32 v30, v71, v51
	v_mul_f32_e32 v31, v69, v51
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v27, v29
	v_fmac_f32_e32 v26, v28, v29
	v_cvt_f32_i32_e32 v27, v52
	v_mul_f32_e32 v20, v30, v20
	v_mul_f32_e32 v21, v31, v21
	v_mul_f32_e32 v28, v72, v51
	v_mul_f32_e32 v30, v70, v51
	v_add_f32_e32 v113, v113, v24
	v_mul_f32_e32 v24, v65, v51
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v20, v28, v27
	v_fmac_f32_e32 v21, v30, v27
	v_add_f32_e32 v114, v114, v25
	v_mul_f32_e32 v25, v67, v51
	v_cvt_f32_i32_e32 v23, v23
	v_add_f32_e32 v111, v111, v20
	v_add_f32_e32 v110, v110, v21
	v_mul_f32_e32 v20, v24, v22
	v_mul_f32_e32 v21, v66, v51
	s_wait_dscnt 0x4
	v_mul_f32_e32 v24, v71, v41
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v22, v25, v23
	v_mul_f32_e32 v23, v68, v51
	v_fmac_f32_e32 v20, v21, v27
	v_cvt_f32_i32_e32 v21, v42
	v_mul_f32_e32 v25, v69, v41
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v16, v24, v16
	v_mul_f32_e32 v24, v72, v41
	v_fmac_f32_e32 v22, v23, v27
	v_add_f32_e32 v109, v109, v20
	v_mul_f32_e32 v17, v25, v17
	v_mul_f32_e32 v20, v70, v41
	v_fmac_f32_e32 v16, v24, v21
	v_mul_f32_e32 v23, v65, v41
	v_mul_f32_e32 v24, v67, v41
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v108, v108, v22
	v_fmac_f32_e32 v17, v20, v21
	v_add_f32_e32 v107, v107, v16
	v_mul_f32_e32 v16, v23, v18
	v_mul_f32_e32 v18, v24, v19
	v_mul_f32_e32 v19, v66, v41
	s_wait_dscnt 0x3
	v_mul_f32_e32 v22, v55, v47
	s_wait_dscnt 0x2
	v_mul_f32_e32 v23, v55, v49
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v106, v106, v17
	v_fmac_f32_e32 v16, v19, v21
	v_mul_f32_e32 v17, v55, v48
	v_mul_f32_e32 v12, v22, v12
	v_mul_f32_e32 v13, v23, v13
	v_mul_f32_e32 v19, v55, v50
	v_mul_f32_e32 v20, v68, v41
	v_cvt_f32_i32_e32 v14, v14
	v_fmac_f32_e32 v12, v17, v56
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v13, v19, v56
	v_fmac_f32_e32 v18, v20, v21
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v55, v45
	v_add_f32_e32 v103, v103, v12
	v_mul_f32_e32 v12, v53, v47
	v_add_f32_e32 v102, v102, v13
	v_mul_f32_e32 v13, v53, v49
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v105, v105, v16
	v_mul_f32_e32 v14, v20, v14
	v_mul_f32_e32 v16, v55, v46
	v_mul_f32_e32 v8, v12, v8
	v_mul_f32_e32 v12, v53, v48
	v_mul_f32_e32 v9, v13, v9
	v_mul_f32_e32 v13, v53, v45
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v14, v16, v56
	v_fmac_f32_e32 v8, v12, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v12, v53, v43
	v_cvt_f32_i32_e32 v11, v11
	v_mul_f32_e32 v10, v13, v10
	v_mul_f32_e32 v13, v53, v46
	v_add_f32_e32 v101, v101, v14
	v_mul_f32_e32 v14, v53, v50
	v_add_f32_e32 v99, v99, v8
	v_mul_f32_e32 v8, v12, v11
	v_mul_f32_e32 v11, v53, v44
	v_fmac_f32_e32 v10, v13, v29
	v_mul_f32_e32 v12, v51, v47
	v_cvt_f32_i32_e32 v4, v4
	v_fmac_f32_e32 v9, v14, v29
	v_fmac_f32_e32 v8, v11, v29
	v_add_f32_e32 v97, v97, v10
	v_mul_f32_e32 v10, v51, v48
	v_mul_f32_e32 v4, v12, v4
	v_mul_f32_e32 v11, v51, v45
	v_mul_f32_e32 v12, v51, v43
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_add_f32_e32 v98, v98, v9
	v_mul_f32_e32 v9, v51, v49
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v4, v10, v27
	v_mul_f32_e32 v6, v11, v6
	v_mul_f32_e32 v7, v12, v7
	v_mul_f32_e32 v10, v51, v46
	v_mul_f32_e32 v11, v51, v44
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v9, v51, v50
	v_add_f32_e32 v95, v95, v4
	v_fmac_f32_e32 v6, v10, v27
	v_fmac_f32_e32 v7, v11, v27
	v_mul_f32_e32 v4, v41, v47
	v_cvt_f32_i32_e32 v0, v0
	v_mul_f32_e32 v22, v55, v43
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v96, v96, v8
	v_fmac_f32_e32 v5, v9, v27
	v_mul_f32_e32 v8, v41, v49
	v_cvt_f32_i32_e32 v1, v1
	v_add_f32_e32 v82, v82, v6
	v_add_f32_e32 v81, v81, v7
	v_mul_f32_e32 v0, v4, v0
	v_mul_f32_e32 v4, v41, v48
	v_mul_f32_e32 v6, v41, v45
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v7, v41, v43
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v22, v15
	v_mul_f32_e32 v17, v55, v44
	v_add_f32_e32 v83, v83, v5
	v_mul_f32_e32 v1, v8, v1
	v_mul_f32_e32 v5, v41, v50
	v_mul_f32_e32 v2, v6, v2
	v_mul_f32_e32 v6, v41, v46
	v_fmac_f32_e32 v0, v4, v21
	v_mul_f32_e32 v3, v7, v3
	v_mul_f32_e32 v4, v41, v44
	v_fmac_f32_e32 v15, v17, v56
	v_fmac_f32_e32 v1, v5, v21
	v_fmac_f32_e32 v2, v6, v21
	v_add_f32_e32 v112, v112, v26
	v_fmac_f32_e32 v3, v4, v21
	v_add_f32_e32 v104, v104, v18
	v_add_f32_e32 v100, v100, v15
	v_add_f32_e32 v77, v77, v0
	v_add_f32_e32 v80, v80, v1
	v_add_f32_e32 v79, v79, v2
	v_add_f32_e32 v78, v78, v3
	s_xor_b64 s[36:37], s[34:35], -1
	s_mov_b32 s56, 1
	s_mov_b64 s[34:35], 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[36:37], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_215
.LBB2_218:                              ;   Parent Loop BB2_216 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s24, s56, s27
	s_lshl_b32 s58, s56, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[24:25], s[42:43]
	s_mov_b32 s59, s25
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[60:61], 0x48
	s_add_nc_u64 s[58:59], s[30:31], s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[60:61], s[50:51], s[60:61]
	v_add_nc_u32_e32 v45, s38, v90
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[62:63], s60, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s61, 0, s[62:63]
	v_add_co_u32 v2, s[62:63], s58, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s59, 0, s[62:63]
	v_add_co_u32 v4, s[62:63], s60, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s61, 0, s[62:63]
	v_add_co_u32 v6, s[60:61], s58, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s59, 0, s[60:61]
	global_load_b64 v[63:64], v[0:1], off offset:40
	global_load_b64 v[57:58], v[2:3], off offset:40
	global_load_b64 v[61:62], v[4:5], off offset:40
	global_load_b64 v[59:60], v[6:7], off offset:40
	v_add_nc_u32_e32 v46, s39, v87
	ds_load_2addr_stride64_b32 v[41:42], v45 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v46 offset0:8 offset1:10
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:12 offset1:14
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v44, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v44, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[41:42], v45 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:9 offset1:11
	v_add_nc_u32_e32 v45, s39, v94
	ds_load_b32 v46, v46 offset:3328
	ds_load_b32 v45, v45
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v43, 0x2000, v91
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v63, s[4:5]
	v_add_nc_u32_e32 v42, 0x4000, v91
	v_cndmask_b32_e64 v44, 0, v64, s[4:5]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v43, v57, v58 offset1:16
	ds_store_2addr_b32 v42, v41, v44 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v61, s[6:7]
	v_cndmask_b32_e64 v42, 0, v62, s[6:7]
	v_add_nc_u32_e32 v43, 0x4000, v93
	v_add_nc_u32_e32 v44, 0x2000, v93
	ds_store_2addr_b32 v43, v41, v42 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v44, v59, v60 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[38:39], s[36:37], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[38:39]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_220
; %bb.219:                              ;   in Loop: Header=BB2_218 Depth=2
	s_add_co_i32 s24, s24, 1
	s_xor_b32 s64, s56, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[24:25], s[42:43]
	s_add_co_i32 s24, s56, s26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[58:59], 0x48
	s_mul_u64 s[60:61], s[24:25], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[58:59], s[50:51], s[58:59]
	s_add_nc_u64 s[56:57], s[48:49], s[60:61]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[45:46], null, 0x48, v88, s[58:59]
	v_add_co_u32 v41, s[62:63], s58, v36
	s_lshl_b32 s60, s64, 6
	s_mov_b32 s61, s25
	s_mulk_i32 s24, 0x88
	v_add_co_ci_u32_e64 v42, null, s59, 0, s[62:63]
	v_add_co_u32 v45, vcc, v45, v84
	v_add_co_u32 v43, s[62:63], s58, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[56:57], s[56:57], s[60:61]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, 0, v46, vcc
	v_add_co_u32 v51, vcc, v33, s24
	v_add_co_ci_u32_e64 v44, null, s59, 0, s[62:63]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v47, s[58:59], s56, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v48, null, s57, 0, s[58:59]
	v_add_co_u32 v49, s[58:59], s56, v39
	s_lshl_b32 s24, s64, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v50, null, s57, 0, s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v51, vcc, v51, s24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v52, vcc
	s_clause 0x1
	global_load_b64 v[63:64], v[41:42], off offset:8
	global_load_b64 v[61:62], v[43:44], off offset:8
	s_clause 0x1
	global_load_b64 v[57:58], v[47:48], off offset:8
	global_load_b64 v[59:60], v[49:50], off offset:8
	global_load_b32 v120, v[45:46], off
	global_load_b32 v121, v[51:52], off
.LBB2_220:                              ; %.preheader508.i.1.i
                                        ;   in Loop: Header=BB2_218 Depth=2
	v_add_nc_u32_e32 v47, 0, v90
	v_add_nc_u32_e32 v48, 0, v87
	s_xor_b64 s[38:39], s[38:39], -1
	s_and_b64 s[56:57], s[34:35], exec
	s_cselect_b32 s24, s41, 0x3400
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:72 offset1:74
	ds_load_2addr_stride64_b32 v[45:46], v48 offset0:76 offset1:78
	s_cselect_b32 s56, s54, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v45, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v46, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v45, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v46, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v45, 0, v94
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:73 offset1:75
	ds_load_b32 v46, v48 offset:19712
	ds_load_b32 v45, v45 offset:16384
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v43, s56, v92
	v_add_nc_u32_e32 v41, s24, v89
	s_and_not1_b64 vcc, exec, s[38:39]
	s_movk_i32 s38, 0x2000
	s_movk_i32 s39, 0x4000
	ds_load_2addr_b32 v[71:72], v43 offset1:1
	ds_load_2addr_b32 v[69:70], v43 offset0:2 offset1:3
	ds_load_2addr_b32 v[65:66], v43 offset0:4 offset1:5
	ds_load_2addr_b32 v[67:68], v43 offset0:6 offset1:7
	ds_load_2addr_b32 v[55:56], v41 offset0:128 offset1:129
	ds_load_2addr_b32 v[53:54], v41 offset0:160 offset1:161
	ds_load_2addr_b32 v[51:52], v41 offset0:192 offset1:193
	ds_load_2addr_b32 v[41:42], v41 offset0:224 offset1:225
	ds_load_2addr_b32 v[47:48], v43 offset0:32 offset1:33
	ds_load_2addr_b32 v[49:50], v43 offset0:34 offset1:35
	ds_load_2addr_b32 v[45:46], v43 offset0:36 offset1:37
	ds_load_2addr_b32 v[43:44], v43 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_217
; %bb.221:                              ; %.preheader509.i.1.i
                                        ;   in Loop: Header=BB2_218 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v122, v86, v121
	s_and_b64 s[36:37], s[36:37], exec
	v_cndmask_b32_e64 v63, 0, v63, s[4:5]
	v_cndmask_b32_e64 v64, 0, v64, s[4:5]
	s_cselect_b32 s24, s54, 0x3c00
	v_cvt_f32_f16_e32 v122, v122.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v123, s24, v85
	ds_store_b32 v91, v57 offset:4096
	ds_store_2addr_b32 v91, v63, v64 offset1:16
	v_cndmask_b32_e64 v57, 0, v61, s[6:7]
	v_cndmask_b32_e64 v61, 0, v62, s[6:7]
	v_cndmask_b32_e64 v122, 0, v122, s[2:3]
	s_cselect_b32 s24, s41, 0x3400
	s_mov_b32 s39, 0
	s_movk_i32 s38, 0x1000
	v_cndmask_b32_e64 v62, 0, v120, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v63, s24, v85
	ds_store_b32 v91, v58 offset:4160
	ds_store_b32 v93, v59 offset:4096
	ds_store_2addr_b32 v93, v57, v61 offset1:16
	ds_store_b32 v93, v60 offset:4160
	ds_store_b32 v63, v62
	ds_store_b32 v123, v122
	s_branch .LBB2_217
.LBB2_222:                              ; %._crit_edge580.i.1.i
	ds_store_2addr_b32 v75, v119, v118 offset1:20
	ds_store_2addr_b32 v75, v117, v116 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v3, 64, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[0:1], null, s40, v3, 0
	v_cmp_gt_i32_e64 s[0:1], s42, v3
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v32
	v_add_co_u32 v0, s[2:3], s52, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s53, v1, s[2:3]
	s_mov_b64 s[2:3], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_226
; %bb.223:
	s_and_b64 s[4:5], s[8:9], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB2_225
; %bb.224:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off
.LBB2_225:                              ; %Flow1065
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB2_226:                              ; %Flow1066
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_lshl_b32 s26, s40, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_228
; %bb.227:
	buffer_load_b32 v3, v73, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v73, s[44:47], s26 offen
.LBB2_228:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_232
; %bb.229:
	s_and_b64 s[4:5], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB2_231
; %bb.230:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:128
.LBB2_231:                              ; %Flow1063
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB2_232:                              ; %Flow1064
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_234
; %bb.233:
	s_or_b32 s2, s26, 0x80
	buffer_load_b32 v3, v73, s[44:47], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB2_234:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_238
; %bb.235:
	s_and_b64 s[4:5], s[12:13], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB2_237
; %bb.236:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:256
.LBB2_237:                              ; %Flow1061
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB2_238:                              ; %Flow1062
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_240
; %bb.239:
	s_add_co_i32 s2, s26, 0x100
	buffer_load_b32 v3, v73, s[44:47], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB2_240:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_244
; %bb.241:
	s_and_b64 s[4:5], s[14:15], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB2_243
; %bb.242:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	global_load_b32 v5, v[3:4], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v2, v5
	global_store_b32 v[3:4], v5, off offset:384
.LBB2_243:                              ; %Flow1059
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB2_244:                              ; %Flow1060
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_246
; %bb.245:
	s_add_co_i32 s2, s26, 0x180
	buffer_load_b32 v3, v73, s[44:47], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB2_246:                              ; %.preheader.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, 0x50, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[2:3], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v115, v114 offset1:20
	ds_store_2addr_b32 v75, v113, v112 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[4:5], s52, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s53, v3, s[4:5]
	s_mov_b64 s[4:5], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_250
; %bb.247:
	s_and_b64 s[6:7], s[8:9], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB2_249
; %bb.248:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off
.LBB2_249:                              ; %Flow1057
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB2_250:                              ; %Flow1058
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_mul_i32 s27, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_252
; %bb.251:
	buffer_load_b32 v5, v73, s[44:47], s27 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s27 offen
.LBB2_252:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB2_256
; %bb.253:
	s_and_b64 s[6:7], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB2_255
; %bb.254:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:128
.LBB2_255:                              ; %Flow1055
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB2_256:                              ; %Flow1056
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_258
; %bb.257:
	s_add_co_i32 s4, s27, 0x80
	buffer_load_b32 v5, v73, s[44:47], s4 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB2_258:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB2_262
; %bb.259:
	s_and_b64 s[6:7], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB2_261
; %bb.260:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:256
.LBB2_261:                              ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB2_262:                              ; %Flow1054
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_264
; %bb.263:
	s_add_co_i32 s4, s27, 0x100
	buffer_load_b32 v5, v73, s[44:47], s4 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB2_264:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB2_268
; %bb.265:
	s_and_b64 s[6:7], s[14:15], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB2_267
; %bb.266:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	global_load_b32 v7, v[5:6], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v4, v7
	global_store_b32 v[5:6], v7, off offset:384
.LBB2_267:                              ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB2_268:                              ; %Flow1052
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_270
; %bb.269:
	s_add_co_i32 s4, s27, 0x180
	buffer_load_b32 v5, v73, s[44:47], s4 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB2_270:                              ; %.preheader.2.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 0x60, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[4:5], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v111, v110 offset1:20
	ds_store_2addr_b32 v75, v109, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[6:7], s52, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s53, v5, s[6:7]
	s_mov_b64 s[6:7], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_274
; %bb.271:
	s_and_b64 s[24:25], s[8:9], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB2_273
; %bb.272:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB2_273:                              ; %Flow1049
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB2_274:                              ; %Flow1050
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_mul_i32 s28, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_276
; %bb.275:
	buffer_load_b32 v7, v73, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s28 offen
.LBB2_276:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB2_280
; %bb.277:
	s_and_b64 s[24:25], s[10:11], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB2_279
; %bb.278:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB2_279:                              ; %Flow1047
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB2_280:                              ; %Flow1048
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_282
; %bb.281:
	s_add_co_i32 s6, s28, 0x80
	buffer_load_b32 v7, v73, s[44:47], s6 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB2_282:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB2_286
; %bb.283:
	s_and_b64 s[24:25], s[12:13], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB2_285
; %bb.284:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB2_285:                              ; %Flow1045
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB2_286:                              ; %Flow1046
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_288
; %bb.287:
	s_add_co_i32 s6, s28, 0x100
	buffer_load_b32 v7, v73, s[44:47], s6 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB2_288:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB2_292
; %bb.289:
	s_and_b64 s[24:25], s[14:15], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB2_291
; %bb.290:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB2_291:                              ; %Flow1043
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB2_292:                              ; %Flow1044
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_294
; %bb.293:
	s_add_co_i32 s6, s28, 0x180
	buffer_load_b32 v7, v73, s[44:47], s6 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB2_294:                              ; %.preheader.3.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 0x70, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[6:7], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v107, v106 offset1:20
	ds_store_2addr_b32 v75, v105, v104 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[24:25], s52, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s53, v7, s[24:25]
	s_mov_b64 s[24:25], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_298
; %bb.295:
	s_and_b64 s[24:25], s[8:9], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_cbranch_execz .LBB2_297
; %bb.296:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB2_297:                              ; %Flow1041
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[24:25], 0
.LBB2_298:                              ; %Flow1042
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[24:25]
	s_mul_i32 s24, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_300
; %bb.299:
	buffer_load_b32 v9, v73, s[44:47], s24 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s24 offen
.LBB2_300:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_304
; %bb.301:
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_303
; %bb.302:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:128
.LBB2_303:                              ; %Flow1039
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_304:                              ; %Flow1040
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_306
; %bb.305:
	s_add_co_i32 s8, s24, 0x80
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_306:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_310
; %bb.307:
	s_and_b64 s[10:11], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_309
; %bb.308:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:256
.LBB2_309:                              ; %Flow1037
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_310:                              ; %Flow1038
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_312
; %bb.311:
	s_add_co_i32 s8, s24, 0x100
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_312:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_316
; %bb.313:
	s_and_b64 s[10:11], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_315
; %bb.314:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:384
.LBB2_315:                              ; %Flow1035
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_316:                              ; %Flow1036
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_318
; %bb.317:
	s_add_co_i32 s8, s24, 0x180
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_318:                              ; %.preheader505.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v103, v102 offset1:20
	ds_store_2addr_b32 v75, v101, v100 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_322
; %bb.319:
	s_and_b64 s[10:11], s[16:17], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_321
; %bb.320:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:64
.LBB2_321:                              ; %Flow1033
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_322:                              ; %Flow1034
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_324
; %bb.323:
	s_or_b32 s8, s26, 64
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_324:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_328
; %bb.325:
	s_and_b64 s[10:11], s[18:19], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_327
; %bb.326:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:192
.LBB2_327:                              ; %Flow1031
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_328:                              ; %Flow1032
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_330
; %bb.329:
	s_or_b32 s8, s26, 0xc0
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_330:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_334
; %bb.331:
	s_and_b64 s[10:11], s[20:21], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB2_333
; %bb.332:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off offset:320
.LBB2_333:                              ; %Flow1029
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB2_334:                              ; %Flow1030
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_336
; %bb.335:
	s_add_co_i32 s8, s26, 0x140
	buffer_load_b32 v9, v73, s[44:47], s8 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB2_336:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB2_340
; %bb.337:
	s_and_b64 s[8:9], s[22:23], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB2_339
; %bb.338:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	global_load_b32 v9, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v8, v9
	global_store_b32 v[0:1], v9, off offset:448
.LBB2_339:                              ; %Flow1027
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[8:9], 0
.LBB2_340:                              ; %Flow1028
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_342
; %bb.341:
	s_addk_co_i32 s26, 0x1c0
	buffer_load_b32 v0, v73, s[44:47], s26 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v8, v0
	buffer_store_b32 v0, v73, s[44:47], s26 offen
.LBB2_342:                              ; %.preheader.1.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v99, v98 offset1:20
	ds_store_2addr_b32 v75, v97, v96 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_346
; %bb.343:
	s_and_b64 s[8:9], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB2_345
; %bb.344:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:64
.LBB2_345:                              ; %Flow1025
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_346:                              ; %Flow1026
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_348
; %bb.347:
	s_add_co_i32 s0, s27, 64
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_348:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_352
; %bb.349:
	s_and_b64 s[8:9], s[18:19], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB2_351
; %bb.350:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:192
.LBB2_351:                              ; %Flow1023
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_352:                              ; %Flow1024
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_354
; %bb.353:
	s_add_co_i32 s0, s27, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_354:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_358
; %bb.355:
	s_and_b64 s[8:9], s[20:21], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB2_357
; %bb.356:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	global_load_b32 v1, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v0, v1
	global_store_b32 v[8:9], v1, off offset:320
.LBB2_357:                              ; %Flow1021
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_358:                              ; %Flow1022
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_360
; %bb.359:
	s_add_co_i32 s0, s27, 0x140
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_360:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_364
; %bb.361:
	s_and_b64 s[2:3], s[22:23], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_363
; %bb.362:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_363:                              ; %Flow1019
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_364:                              ; %Flow1020
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_366
; %bb.365:
	s_addk_co_i32 s27, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s27 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s27 offen
.LBB2_366:                              ; %.preheader.2.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v95, v83 offset1:20
	ds_store_2addr_b32 v75, v82, v81 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_370
; %bb.367:
	s_and_b64 s[2:3], s[16:17], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_369
; %bb.368:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB2_369:                              ; %Flow1017
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_370:                              ; %Flow1018
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_372
; %bb.371:
	s_or_b32 s0, s28, 64
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_372:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_376
; %bb.373:
	s_and_b64 s[2:3], s[18:19], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_375
; %bb.374:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB2_375:                              ; %Flow1015
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_376:                              ; %Flow1016
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_378
; %bb.377:
	s_add_co_i32 s0, s28, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_378:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_382
; %bb.379:
	s_and_b64 s[2:3], s[20:21], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_381
; %bb.380:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB2_381:                              ; %Flow1013
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_382:                              ; %Flow1014
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_384
; %bb.383:
	s_add_co_i32 s0, s28, 0x140
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_384:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_388
; %bb.385:
	s_and_b64 s[2:3], s[22:23], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_387
; %bb.386:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_387:                              ; %Flow1011
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_388:                              ; %Flow1012
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_390
; %bb.389:
	s_addk_co_i32 s28, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s28 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s28 offen
.LBB2_390:                              ; %.preheader.3.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v77, v80 offset1:20
	ds_store_2addr_b32 v75, v79, v78 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_394
; %bb.391:
	s_and_b64 s[2:3], s[16:17], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_393
; %bb.392:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:64
.LBB2_393:                              ; %Flow1009
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_394:                              ; %Flow1010
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_396
; %bb.395:
	s_add_co_i32 s0, s24, 64
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_396:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_400
; %bb.397:
	s_and_b64 s[2:3], s[18:19], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_399
; %bb.398:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:192
.LBB2_399:                              ; %Flow1007
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_400:                              ; %Flow1008
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_402
; %bb.401:
	s_add_co_i32 s0, s24, 0xc0
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_402:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_406
; %bb.403:
	s_and_b64 s[2:3], s[20:21], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_405
; %bb.404:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:320
.LBB2_405:                              ; %Flow1005
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_406:                              ; %Flow1006
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_408
; %bb.407:
	s_add_co_i32 s0, s24, 0x140
	buffer_load_b32 v1, v73, s[44:47], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB2_408:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_412
; %bb.409:
	s_and_b64 s[2:3], s[22:23], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_411
; %bb.410:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v0, v3
	global_store_b32 v[1:2], v3, off offset:448
.LBB2_411:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_412:                              ; %Flow1004
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_414
; %bb.413:
	s_addk_co_i32 s24, 0x1c0
	buffer_load_b32 v1, v73, s[44:47], s24 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v1
	buffer_store_b32 v0, v73, s[44:47], s24 offen
.LBB2_414:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end2:
	.size	iu4_w64_r2_add, .Lfunc_end2-iu4_w64_r2_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r2_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 132
		.amdhsa_next_free_sgpr 65
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-iu4_w64_r2_add)<<4)&4080)>>4
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
	.set .Liu4_w64_r2_add.num_vgpr, 132
	.set .Liu4_w64_r2_add.num_agpr, 0
	.set .Liu4_w64_r2_add.numbered_sgpr, 65
	.set .Liu4_w64_r2_add.num_named_barrier, 0
	.set .Liu4_w64_r2_add.private_seg_size, 0
	.set .Liu4_w64_r2_add.uses_vcc, 1
	.set .Liu4_w64_r2_add.uses_flat_scratch, 0
	.set .Liu4_w64_r2_add.has_dyn_sized_stack, 0
	.set .Liu4_w64_r2_add.has_recursion, 0
	.set .Liu4_w64_r2_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18924
; TotalNumSgprs: 67
; NumVgprs: 132
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 32
; NumSGPRsForWavesPerEU: 67
; NumVGPRsForWavesPerEU: 132
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	iu4_w64_r2_set          ; -- Begin function iu4_w64_r2_set
	.globl	iu4_w64_r2_set
	.p2align	8
	.type	iu4_w64_r2_set,@function
iu4_w64_r2_set:                         ; @iu4_w64_r2_set
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[40:42], s[0:1], 0x18
	s_lshl_b32 s8, ttmp9, 7
	s_lshl_b32 s10, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s8, s40
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_ge_i32 s10, s42
	s_cselect_b64 s[4:5], -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_3
; %bb.1:                                ; %Flow1141
	s_and_not1_b64 vcc, exec, s[2:3]
	s_cbranch_vccz .LBB3_4
.LBB3_2:                                ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB3_3:                                ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb0ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.1.critedge.i
	s_cbranch_execnz .LBB3_2
.LBB3_4:                                ; %.preheader509.i.i
	s_clause 0x1
	s_load_b128 s[48:51], s[0:1], 0x0
	s_load_b64 s[52:53], s[0:1], 0x10
	v_lshrrev_b32_e32 v1, 1, v0
	s_ashr_i32 s0, s41, 31
	v_and_b32_e32 v5, 1, v0
	s_lshr_b32 s0, s0, 24
	s_add_co_i32 s4, s40, -1
	v_or_b32_e32 v6, s10, v1
	v_or_b32_e32 v7, s8, v1
	s_add_co_i32 s0, s41, s0
	v_lshlrev_b32_e32 v84, 2, v5
	s_ashr_i32 s33, s0, 8
	v_lshlrev_b32_e32 v8, 3, v1
	v_min_i32_e32 v9, s4, v7
	s_mul_i32 s6, s33, 0x88
	v_cmp_gt_u32_e64 s[24:25], 0x100, v0
	v_mov_b32_e32 v2, 0
	v_cmp_gt_i32_e64 s[28:29], s42, v6
	v_add3_u32 v85, 0, v8, v84
	v_cmp_gt_i32_e64 s[26:27], s40, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[3:4], null, 0x48, v6, s[50:51]
	v_mad_co_i64_i32 v[42:43], null, s6, v9, s[48:49]
	v_lshlrev_b32_e32 v86, 4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v44, vcc, v3, v84
	v_add_co_ci_u32_e64 v45, null, 0, v4, vcc
	s_and_saveexec_b64 s[0:1], s[24:25]
	s_cbranch_execz .LBB3_8
; %bb.5:                                ; %.lr.ph.i.i
	s_and_saveexec_b64 s[2:3], s[28:29]
	s_cbranch_execz .LBB3_7
; %bb.6:
	global_load_b32 v2, v[44:45], off
.LBB3_7:                                ; %.preheader506.loopexit.i.i
	s_or_b64 exec, exec, s[2:3]
	global_load_b32 v3, v[42:43], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v3, v86, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v3, v3.l
	v_cndmask_b32_e64 v3, 0, v3, s[26:27]
	ds_store_2addr_stride64_b32 v85, v2, v3 offset0:48 offset1:56
.LBB3_8:                                ; %Flow1140
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v10, 2, v0
	v_and_b32_e32 v2, 3, v0
	s_add_co_i32 s2, s42, -1
	v_and_b32_e32 v13, 0x60, v1
	v_add_nc_u32_e32 v18, s10, v1
	v_add_nc_u32_e32 v11, s10, v10
	v_add_nc_u32_e32 v3, s8, v10
	v_lshlrev_b32_e32 v2, 3, v2
	v_add_nc_u32_e32 v1, s8, v1
	v_lshrrev_b32_e32 v17, 5, v0
	v_add_nc_u32_e32 v12, 64, v11
	v_add_nc_u32_e32 v4, 64, v3
	v_min_i32_e32 v5, s2, v11
	v_min_i32_e32 v3, s4, v3
	v_bfe_u32 v14, v0, 1, 1
	v_min_i32_e32 v6, s2, v12
	v_min_i32_e32 v4, s4, v4
	v_lshlrev_b32_e32 v15, 7, v0
	v_mad_co_u64_u32 v[36:37], null, 0x48, v5, v[2:3]
	v_mad_co_u64_u32 v[37:38], null, s6, v3, v[2:3]
	v_mad_co_u64_u32 v[38:39], null, 0x48, v6, v[2:3]
	v_mad_co_u64_u32 v[39:40], null, s6, v4, v[2:3]
	v_and_b32_e32 v16, 60, v0
	v_and_b32_e32 v123, 12, v10
	v_min_i32_e32 v10, s4, v1
	v_or_b32_e32 v19, 8, v17
	global_load_b64 v[2:3], v36, s[50:51] offset:8
	global_load_b64 v[4:5], v37, s[48:49] offset:8
	global_load_b64 v[6:7], v38, s[50:51] offset:8
	global_load_b64 v[8:9], v39, s[48:49] offset:8
	v_and_or_b32 v15, 0x80, v15, v16
	v_and_or_b32 v16, v17, 6, v14
	v_mad_co_u64_u32 v[33:34], null, s6, v10, s[48:49]
	v_min_i32_e32 v88, s2, v18
	v_cmp_gt_i32_e64 s[2:3], s40, v1
	v_and_or_b32 v1, v19, 14, v14
	v_ashrrev_i32_e32 v10, 31, v10
	v_lshl_or_b32 v14, v16, 8, v15
	v_cmp_gt_i32_e64 s[4:5], s42, v11
	v_lshlrev_b32_e32 v94, 2, v0
	v_lshl_or_b32 v1, v1, 8, v15
	v_mad_co_u64_u32 v[40:41], null, s6, v10, v[34:35]
	v_cmp_gt_i32_e64 s[6:7], s42, v12
	v_add_nc_u32_e32 v91, 0, v14
	v_add_co_u32 v50, s[14:15], s50, v36
	v_add_nc_u32_e32 v93, 0, v1
	v_add_co_ci_u32_e64 v51, null, s51, 0, s[14:15]
	v_add_co_u32 v52, s[14:15], s48, v37
	v_and_b32_e32 v124, 15, v0
	v_and_b32_e32 v87, 0xfc, v94
	v_or_b32_e32 v16, v123, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v53, null, s49, 0, s[14:15]
	v_add_co_u32 v46, s[14:15], s50, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v47, null, s51, 0, s[14:15]
	v_add_co_u32 v48, s[14:15], s48, v39
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v98, 0
	s_cmp_gt_i32 s41, 0xff
	v_mov_b32_e32 v95, 0
	v_cmp_gt_i32_e64 s[0:1], s42, v18
	v_lshl_add_u32 v89, v124, 3, 0
	v_lshl_or_b32 v90, v13, 5, v87
	v_lshl_add_u32 v92, v16, 3, 0
	v_add_co_ci_u32_e64 v49, null, s49, 0, s[14:15]
	s_cselect_b64 s[54:55], -1, 0
	s_ashr_i32 s43, s42, 31
	s_mov_b32 s13, 0
	s_cmp_lt_i32 s41, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s[4:5]
	v_cndmask_b32_e64 v3, 0, v3, s[4:5]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v1, 0, v6, s[6:7]
	v_cndmask_b32_e64 v6, 0, v7, s[6:7]
	ds_store_b32 v91, v4 offset:4096
	ds_store_2addr_b32 v91, v2, v3 offset1:16
	ds_store_b32 v91, v5 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v93, v8 offset:4096
	ds_store_2addr_b32 v93, v1, v6 offset1:16
	ds_store_b32 v93, v9 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_17
; %bb.9:                                ; %.preheader505.lr.ph.i.i
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	s_movk_i32 s30, 0x1000
	s_movk_i32 s9, 0x3000
	s_movk_i32 s11, 0x3800
	s_mov_b32 s31, 0
	s_mov_b32 s14, s13
	s_branch .LBB3_11
.LBB3_10:                               ;   in Loop: Header=BB3_11 Depth=1
	s_and_b64 vcc, exec, s[16:17]
	s_mov_b32 s14, s34
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_17
.LBB3_11:                               ; %.preheader505.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_13 Depth 2
	s_mov_b32 s15, s13
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s34, s14, 1
	s_mul_u64 s[18:19], s[14:15], 0x88
	s_lshl_b32 s15, s14, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s34, s33
	s_mov_b32 s35, 0
	s_cselect_b64 s[16:17], -1, 0
	s_add_nc_u64 s[18:19], s[48:49], s[18:19]
	s_mov_b64 s[22:23], 0
	s_mov_b64 s[20:21], -1
	s_branch .LBB3_13
.LBB3_12:                               ;   in Loop: Header=BB3_13 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v68, v82, v66
	v_mul_f32_e32 v69, v80, v66
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v67, v67
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v70, v78, v66
	v_cvt_f32_i32_e32 v31, v31
	v_mul_f32_e32 v29, v68, v29
	v_mul_f32_e32 v30, v69, v30
	v_mul_f32_e32 v68, v83, v66
	v_mul_f32_e32 v69, v81, v66
	v_cvt_f32_i32_e32 v32, v32
	v_mul_f32_e32 v31, v70, v31
	v_cvt_f32_i32_e32 v25, v25
	v_fmac_f32_e32 v29, v68, v67
	v_fmac_f32_e32 v30, v69, v67
	v_mul_f32_e32 v68, v76, v66
	v_mul_f32_e32 v69, v79, v66
	v_cvt_f32_i32_e32 v26, v26
	v_add_f32_e32 v128, v128, v29
	v_add_f32_e32 v127, v127, v30
	v_mul_f32_e32 v29, v68, v32
	v_mul_f32_e32 v30, v77, v66
	v_fmac_f32_e32 v31, v69, v67
	s_wait_dscnt 0x6
	v_mul_f32_e32 v32, v82, v64
	v_mul_f32_e32 v68, v80, v64
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v29, v30, v67
	v_add_f32_e32 v126, v126, v31
	v_cvt_f32_i32_e32 v30, v65
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v26, v68, v26
	v_mul_f32_e32 v31, v83, v64
	v_mul_f32_e32 v32, v81, v64
	v_add_f32_e32 v125, v125, v29
	v_mul_f32_e32 v29, v78, v64
	v_cvt_f32_i32_e32 v28, v28
	v_fmac_f32_e32 v25, v31, v30
	v_fmac_f32_e32 v26, v32, v30
	v_mul_f32_e32 v31, v76, v64
	v_mul_f32_e32 v27, v29, v27
	v_mul_f32_e32 v29, v79, v64
	v_add_f32_e32 v121, v121, v25
	v_add_f32_e32 v122, v122, v26
	v_mul_f32_e32 v25, v31, v28
	v_mul_f32_e32 v26, v77, v64
	v_fmac_f32_e32 v27, v29, v30
	s_wait_dscnt 0x5
	v_mul_f32_e32 v28, v82, v62
	v_mul_f32_e32 v29, v80, v62
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v25, v26, v30
	v_add_f32_e32 v119, v119, v27
	v_cvt_f32_i32_e32 v26, v63
	v_mul_f32_e32 v21, v28, v21
	v_mul_f32_e32 v22, v29, v22
	v_mul_f32_e32 v27, v78, v62
	v_cvt_f32_i32_e32 v23, v23
	v_mul_f32_e32 v28, v83, v62
	v_mul_f32_e32 v29, v81, v62
	v_mul_f32_e32 v31, v76, v62
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v23, v27, v23
	v_mul_f32_e32 v27, v79, v62
	v_fmac_f32_e32 v21, v28, v26
	v_fmac_f32_e32 v22, v29, v26
	v_mul_f32_e32 v24, v31, v24
	v_mul_f32_e32 v28, v77, v62
	v_fmac_f32_e32 v23, v27, v26
	v_add_f32_e32 v117, v117, v21
	v_add_f32_e32 v118, v118, v22
	s_wait_dscnt 0x4
	v_mul_f32_e32 v21, v82, v34
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v80, v34
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v24, v28, v26
	v_add_f32_e32 v116, v116, v23
	v_cvt_f32_i32_e32 v23, v35
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v83, v34
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v78, v34
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v115, v115, v24
	v_mul_f32_e32 v24, v81, v34
	v_fmac_f32_e32 v17, v21, v23
	v_mul_f32_e32 v21, v76, v34
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v79, v34
	v_fmac_f32_e32 v18, v24, v23
	v_add_f32_e32 v113, v113, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v77, v34
	v_fmac_f32_e32 v19, v22, v23
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v66, v60
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v66, v58
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v114, v114, v18
	v_fmac_f32_e32 v17, v20, v23
	v_add_f32_e32 v111, v111, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v66, v61
	v_mul_f32_e32 v19, v66, v59
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v66, v54
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v112, v112, v17
	v_fmac_f32_e32 v13, v18, v67
	v_fmac_f32_e32 v14, v19, v67
	v_mul_f32_e32 v17, v66, v55
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v64, v60
	v_mul_f32_e32 v20, v64, v58
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v109, v109, v13
	v_fmac_f32_e32 v15, v17, v67
	v_mul_f32_e32 v13, v64, v61
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v64, v59
	v_add_f32_e32 v110, v110, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v30
	v_mul_f32_e32 v13, v64, v54
	v_fmac_f32_e32 v10, v17, v30
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v64, v56
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v105, v105, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v106, v106, v10
	v_mul_f32_e32 v10, v64, v55
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v62, v60
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v30
	v_mul_f32_e32 v10, v62, v58
	v_mul_f32_e32 v13, v64, v57
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v62, v61
	v_add_f32_e32 v104, v104, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v62, v59
	v_mul_f32_e32 v10, v62, v54
	v_fmac_f32_e32 v5, v12, v26
	v_mul_f32_e32 v12, v62, v56
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v30
	v_fmac_f32_e32 v6, v9, v26
	v_add_f32_e32 v101, v101, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v62, v55
	v_mul_f32_e32 v9, v62, v57
	v_mul_f32_e32 v21, v66, v56
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v103, v103, v11
	v_mul_f32_e32 v10, v34, v60
	v_mul_f32_e32 v11, v34, v58
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v26
	v_fmac_f32_e32 v7, v9, v26
	v_mul_f32_e32 v8, v34, v54
	v_mul_f32_e32 v9, v34, v56
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v66, v57
	v_add_f32_e32 v102, v102, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v34, v61
	v_mul_f32_e32 v10, v34, v59
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v34, v55
	v_mul_f32_e32 v9, v34, v57
	v_fmac_f32_e32 v16, v18, v67
	v_fmac_f32_e32 v1, v6, v23
	v_fmac_f32_e32 v2, v10, v23
	v_fmac_f32_e32 v3, v8, v23
	v_fmac_f32_e32 v4, v9, v23
	v_add_f32_e32 v120, v120, v25
	v_add_f32_e32 v107, v107, v15
	v_add_f32_e32 v108, v108, v16
	v_add_f32_e32 v99, v99, v5
	v_add_f32_e32 v100, v100, v7
	v_add_f32_e32 v95, v95, v1
	v_add_f32_e32 v98, v98, v2
	v_add_f32_e32 v96, v96, v3
	v_add_f32_e32 v97, v97, v4
	s_xor_b64 s[22:23], s[20:21], -1
	s_mov_b32 s35, 1
	s_mov_b64 s[20:21], 0
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[22:23], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_10
.LBB3_13:                               ;   Parent Loop BB3_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s12, s35, s15
	s_lshl_b32 s36, s35, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[12:13], s[42:43]
	s_mov_b32 s37, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[38:39], 0x48
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[50:51], s[38:39]
	v_add_nc_u32_e32 v56, s30, v90
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[44:45], s38, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s39, 0, s[44:45]
	v_add_co_u32 v3, s[44:45], s36, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s37, 0, s[44:45]
	v_add_co_u32 v5, s[44:45], s38, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s39, 0, s[44:45]
	v_add_co_u32 v7, s[38:39], s36, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s37, 0, s[38:39]
	global_load_b64 v[74:75], v[1:2], off offset:40
	global_load_b64 v[68:69], v[3:4], off offset:40
	global_load_b64 v[72:73], v[5:6], off offset:40
	global_load_b64 v[70:71], v[7:8], off offset:40
	v_add_nc_u32_e32 v57, s31, v87
	ds_load_2addr_stride64_b32 v[34:35], v56 offset1:2
	ds_load_2addr_stride64_b32 v[1:2], v57 offset1:2
	ds_load_2addr_stride64_b32 v[54:55], v57 offset0:4 offset1:6
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v2, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v54, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v55, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v2, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v54, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v55, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[34:35], v56 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[54:55], v57 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[56:57], v57 offset0:5 offset1:7
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v54, 0x2000, v91
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v74, s[4:5]
	v_add_nc_u32_e32 v35, 0x4000, v91
	v_cndmask_b32_e64 v55, 0, v75, s[4:5]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v54, v68, v69 offset1:16
	ds_store_2addr_b32 v35, v34, v55 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v72, s[6:7]
	v_cndmask_b32_e64 v35, 0, v73, s[6:7]
	v_add_nc_u32_e32 v54, 0x4000, v93
	v_add_nc_u32_e32 v55, 0x2000, v93
	ds_store_2addr_b32 v54, v34, v35 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v55, v70, v71 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[30:31], s[22:23], s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[30:31]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_15
; %bb.14:                               ;   in Loop: Header=BB3_13 Depth=2
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[12:13], s[42:43]
	s_add_co_i32 s12, s35, s14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_mul_u64 s[38:39], s[12:13], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[50:51], s[36:37]
	s_xor_b32 s35, s35, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[56:57], null, 0x48, v88, s[36:37]
	v_add_co_u32 v34, s[44:45], s36, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, s37, 0, s[44:45]
	s_add_nc_u64 s[38:39], s[48:49], s[38:39]
	s_lshl_b32 s44, s35, 6
	s_mov_b32 s45, s13
	v_add_co_u32 v56, vcc, v56, v84
	s_mulk_i32 s12, 0x88
	v_add_co_u32 v54, s[46:47], s36, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[38:39], s[44:45]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v57, null, 0, v57, vcc
	v_add_co_u32 v41, vcc, v33, s12
	v_add_co_ci_u32_e64 v55, null, s37, 0, s[46:47]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v58, s[36:37], s38, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v59, null, s39, 0, s[36:37]
	v_add_co_u32 v60, s[36:37], s38, v39
	s_lshl_b32 s12, s35, 2
	v_add_co_ci_u32_e64 v61, null, s39, 0, s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v62, vcc, v41, s12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v63, null, 0, v63, vcc
	s_clause 0x1
	global_load_b64 v[74:75], v[34:35], off offset:8
	global_load_b64 v[72:73], v[54:55], off offset:8
	s_clause 0x1
	global_load_b64 v[68:69], v[58:59], off offset:8
	global_load_b64 v[70:71], v[60:61], off offset:8
	global_load_b32 v41, v[56:57], off
	global_load_b32 v129, v[62:63], off
.LBB3_15:                               ; %.preheader503.i.i
                                        ;   in Loop: Header=BB3_13 Depth=2
	v_add_nc_u32_e32 v58, 0, v90
	v_add_nc_u32_e32 v59, 0, v87
	s_xor_b64 s[30:31], s[30:31], -1
	s_and_b64 s[36:37], s[20:21], exec
	s_cselect_b32 s12, s9, 0x3400
	ds_load_2addr_stride64_b32 v[34:35], v58 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[54:55], v59 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:68 offset1:70
	s_cselect_b32 s35, s11, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[34:35], v58 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[54:55], v59 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[56:57], v59 offset0:69 offset1:71
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[29:32], v34, v54, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v34, v55, v[25:28] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[21:24], v34, v56, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v34, v57, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v35, v54, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v35, v55, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v35, v56, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v35, v57, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v56, s35, v92
	v_add_nc_u32_e32 v34, s12, v89
	s_and_not1_b64 vcc, exec, s[30:31]
	s_movk_i32 s30, 0x2000
	s_movk_i32 s31, 0x4000
	ds_load_2addr_b32 v[82:83], v56 offset1:1
	ds_load_2addr_b32 v[80:81], v56 offset0:2 offset1:3
	ds_load_2addr_b32 v[78:79], v56 offset0:4 offset1:5
	ds_load_2addr_b32 v[76:77], v56 offset0:6 offset1:7
	ds_load_2addr_b32 v[66:67], v34 offset1:1
	ds_load_2addr_b32 v[64:65], v34 offset0:32 offset1:33
	ds_load_2addr_b32 v[62:63], v34 offset0:64 offset1:65
	ds_load_2addr_b32 v[34:35], v34 offset0:96 offset1:97
	ds_load_2addr_b32 v[60:61], v56 offset0:32 offset1:33
	ds_load_2addr_b32 v[58:59], v56 offset0:34 offset1:35
	ds_load_2addr_b32 v[54:55], v56 offset0:36 offset1:37
	ds_load_2addr_b32 v[56:57], v56 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.16:                               ; %.preheader504.i.i
                                        ;   in Loop: Header=BB3_13 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v130, v86, v129
	s_and_b64 s[22:23], s[22:23], exec
	v_cndmask_b32_e64 v74, 0, v74, s[4:5]
	v_cndmask_b32_e64 v75, 0, v75, s[4:5]
	s_cselect_b32 s12, s11, 0x3c00
	v_cvt_f32_f16_e64 v130, v130.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v131, s12, v85
	ds_store_b32 v91, v68 offset:4096
	ds_store_2addr_b32 v91, v74, v75 offset1:16
	v_cndmask_b32_e64 v68, 0, v72, s[6:7]
	v_cndmask_b32_e64 v72, 0, v73, s[6:7]
	v_cndmask_b32_e64 v130, 0, v130, s[2:3]
	s_cselect_b32 s12, s9, 0x3400
	s_mov_b32 s31, 0
	s_movk_i32 s30, 0x1000
	v_cndmask_b32_e64 v73, 0, v41, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s12, v85
	ds_store_b32 v91, v69 offset:4160
	ds_store_b32 v93, v70 offset:4096
	ds_store_2addr_b32 v93, v68, v72 offset1:16
	ds_store_b32 v93, v71 offset:4160
	ds_store_b32 v74, v73
	ds_store_b32 v131, v130
	s_branch .LBB3_12
.LBB3_17:                               ; %._crit_edge575.i.i
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v2, 2, v124
	v_lshrrev_b32_e32 v0, 4, v0
	s_ashr_i32 s11, s10, 31
	s_ashr_i32 s41, s40, 31
	v_mul_u32_u24_e32 v1, 0x500, v1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[12:13], s[40:41], s[10:11]
	v_lshlrev_b32_e32 v4, 2, v0
	v_or_b32_e32 v76, s10, v0
	v_mul_lo_u32 v3, s40, v0
	v_add3_u32 v2, 0, v1, v2
	s_ashr_i32 s9, s8, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[12:13], s[12:13], 2
	s_lshl_b64 s[14:15], s[8:9], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[52:53], s[12:13]
	v_mad_u32_u24 v1, 0x50, v123, v2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[44:45], s[12:13], s[14:15]
	s_add_co_i32 s11, s8, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s45, s45, 0xffff
	s_cmp_gt_i32 s11, s40
	ds_store_2addr_b32 v1, v128, v127 offset1:20
	ds_store_2addr_b32 v1, v126, v125 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v124
	s_cselect_b64 s[12:13], -1, 0
	s_add_co_i32 s9, s10, 0x80
	v_or_b32_e32 v34, s8, v124
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s9, s42
	v_add3_u32 v32, 0, v1, v4
	v_mad_co_i64_i32 v[0:1], null, s40, v76, 0
	s_cselect_b64 s[8:9], -1, 0
	v_ashrrev_i32_e32 v35, 31, v34
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[12:13], s[12:13], s[8:9]
	v_cmp_gt_i32_e64 s[8:9], s40, v34
	v_cmp_gt_i32_e64 s[30:31], s42, v76
	s_mov_b32 s47, 0x31004000
	s_mov_b32 s46, -1
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_mov_b64 s[10:11], -1
	v_add_co_u32 v0, vcc, s52, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s53, v1, vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_21
; %bb.18:
	s_and_b64 s[14:15], s[8:9], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[10:11], s[14:15]
	s_cbranch_execz .LBB3_20
; %bb.19:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v1, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off
.LBB3_20:                               ; %Flow1134
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[10:11]
	s_mov_b64 s[10:11], 0
.LBB3_21:                               ; %Flow1135
	v_add_lshl_u32 v73, v3, v124, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_23
; %bb.22:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], null offen
.LBB3_23:
	ds_load_b32 v3, v32 offset:1280
	s_wait_dscnt 0x1
	v_or_b32_e32 v4, 32, v34
	v_cndmask_b32_e64 v74, 0, 1, s[12:13]
	s_and_not1_b64 vcc, exec, s[12:13]
	s_mov_b64 s[12:13], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[10:11], s40, v4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_27
; %bb.24:
	s_and_b64 s[14:15], s[10:11], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB3_26
; %bb.25:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v3, off offset:128
.LBB3_26:                               ; %Flow1132
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[12:13]
	s_mov_b64 s[12:13], 0
.LBB3_27:                               ; %Flow1133
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[12:13]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_29
; %bb.28:
	s_movk_i32 s12, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v3, v73, s[44:47], s12 offen
.LBB3_29:
	s_wait_dscnt 0x0
	ds_load_b32 v3, v32 offset:2560
	v_or_b32_e32 v4, 64, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[14:15], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[12:13], s40, v4
	s_cbranch_vccnz .LBB3_33
; %bb.30:
	s_and_b64 s[16:17], s[12:13], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB3_32
; %bb.31:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v3, off offset:256
.LBB3_32:                               ; %Flow1130
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[14:15]
	s_mov_b64 s[14:15], 0
.LBB3_33:                               ; %Flow1131
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_35
; %bb.34:
	s_movk_i32 s14, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v3, v73, s[44:47], s14 offen
.LBB3_35:
	s_wait_dscnt 0x0
	ds_load_b32 v3, v32 offset:3840
	v_or_b32_e32 v4, 0x60, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[14:15], s40, v4
	s_cbranch_vccnz .LBB3_39
; %bb.36:
	s_and_b64 s[18:19], s[14:15], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_38
; %bb.37:
	v_lshlrev_b64_e32 v[4:5], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v1, v5, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v3, off offset:384
.LBB3_38:                               ; %Flow1128
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_39:                               ; %Flow1129
	v_mul_u32_u24_e32 v4, 0x50, v123
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_41
; %bb.40:
	s_movk_i32 s16, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v3, v73, s[44:47], s16 offen
.LBB3_41:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v75, v2, v4
	v_or_b32_e32 v5, 16, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[34:35], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v121, v122 offset1:20
	ds_store_2addr_b32 v75, v119, v120 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[16:17], s52, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s53, v3, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_45
; %bb.42:
	s_and_b64 s[18:19], s[8:9], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_44
; %bb.43:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off
.LBB3_44:                               ; %Flow1126
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_45:                               ; %Flow1127
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_lshl_b32 s41, s40, 6
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_47
; %bb.46:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s41 offen
.LBB3_47:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_51
; %bb.48:
	s_and_b64 s[18:19], s[10:11], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_50
; %bb.49:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:128
.LBB3_50:                               ; %Flow1124
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_51:                               ; %Flow1125
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_53
; %bb.52:
	s_add_co_i32 s16, s41, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB3_53:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_57
; %bb.54:
	s_and_b64 s[18:19], s[12:13], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_56
; %bb.55:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:256
.LBB3_56:                               ; %Flow1122
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_57:                               ; %Flow1123
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_59
; %bb.58:
	s_add_co_i32 s16, s41, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB3_59:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_63
; %bb.60:
	s_and_b64 s[18:19], s[14:15], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_62
; %bb.61:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:384
.LBB3_62:                               ; %Flow1120
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_63:                               ; %Flow1121
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_65
; %bb.64:
	s_add_co_i32 s16, s41, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s16 offen
.LBB3_65:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 32, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[36:37], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v117, v118 offset1:20
	ds_store_2addr_b32 v75, v116, v115 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[16:17], s52, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s53, v5, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_69
; %bb.66:
	s_and_b64 s[18:19], s[8:9], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_68
; %bb.67:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off
.LBB3_68:                               ; %Flow1118
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_69:                               ; %Flow1119
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_lshl_b32 s58, s40, 7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_71
; %bb.70:
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s58 offen
.LBB3_71:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_75
; %bb.72:
	s_and_b64 s[18:19], s[10:11], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_74
; %bb.73:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:128
.LBB3_74:                               ; %Flow1116
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_75:                               ; %Flow1117
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_77
; %bb.76:
	s_add_co_i32 s16, s58, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB3_77:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_81
; %bb.78:
	s_and_b64 s[18:19], s[12:13], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_80
; %bb.79:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:256
.LBB3_80:                               ; %Flow1114
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_81:                               ; %Flow1115
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_83
; %bb.82:
	s_add_co_i32 s16, s58, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB3_83:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_87
; %bb.84:
	s_and_b64 s[18:19], s[14:15], s[36:37]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_86
; %bb.85:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:384
.LBB3_86:                               ; %Flow1112
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_87:                               ; %Flow1113
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_89
; %bb.88:
	s_add_co_i32 s16, s58, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s16 offen
.LBB3_89:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 48, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[38:39], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v113, v114 offset1:20
	ds_store_2addr_b32 v75, v111, v112 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[16:17], s52, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s53, v7, s[16:17]
	s_mov_b64 s[16:17], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_93
; %bb.90:
	s_and_b64 s[18:19], s[8:9], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_92
; %bb.91:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB3_92:                               ; %Flow1110
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_93:                               ; %Flow1111
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_mul_i32 s59, s40, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_95
; %bb.94:
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s59 offen
.LBB3_95:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_99
; %bb.96:
	s_and_b64 s[18:19], s[10:11], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_98
; %bb.97:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:128
.LBB3_98:                               ; %Flow1108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_99:                               ; %Flow1109
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_101
; %bb.100:
	s_add_co_i32 s16, s59, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB3_101:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_105
; %bb.102:
	s_and_b64 s[18:19], s[12:13], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_104
; %bb.103:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:256
.LBB3_104:                              ; %Flow1106
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_105:                              ; %Flow1107
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_107
; %bb.106:
	s_add_co_i32 s16, s59, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB3_107:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[16:17], -1
	s_cbranch_vccnz .LBB3_111
; %bb.108:
	s_and_b64 s[18:19], s[14:15], s[38:39]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB3_110
; %bb.109:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:384
.LBB3_110:                              ; %Flow1104
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[16:17]
	s_mov_b64 s[16:17], 0
.LBB3_111:                              ; %Flow1105
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[16:17]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_113
; %bb.112:
	s_add_co_i32 s16, s59, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s16 offen
.LBB3_113:                              ; %.preheader500.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 16, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[18:19], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[16:17], s40, v9
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v109, v110 offset1:20
	ds_store_2addr_b32 v75, v107, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_117
; %bb.114:
	s_and_b64 s[20:21], s[16:17], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB3_116
; %bb.115:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:64
.LBB3_116:                              ; %Flow1102
	s_or_b64 exec, exec, s[18:19]
	s_mov_b64 s[18:19], 0
.LBB3_117:                              ; %Flow1103
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_119
; %bb.118:
	s_mov_b32 s18, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s18 offen
.LBB3_119:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_or_b32_e32 v9, 48, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[20:21], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[18:19], s40, v9
	s_cbranch_vccnz .LBB3_123
; %bb.120:
	s_and_b64 s[22:23], s[18:19], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB3_122
; %bb.121:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:192
.LBB3_122:                              ; %Flow1100
	s_or_b64 exec, exec, s[20:21]
	s_mov_b64 s[20:21], 0
.LBB3_123:                              ; %Flow1101
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[20:21]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_125
; %bb.124:
	s_movk_i32 s20, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s20 offen
.LBB3_125:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_or_b32_e32 v9, 0x50, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[22:23], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[20:21], s40, v9
	s_cbranch_vccnz .LBB3_129
; %bb.126:
	s_and_b64 s[56:57], s[20:21], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[22:23], s[56:57]
	s_cbranch_execz .LBB3_128
; %bb.127:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:320
.LBB3_128:                              ; %Flow1098
	s_or_b64 exec, exec, s[22:23]
	s_mov_b64 s[22:23], 0
.LBB3_129:                              ; %Flow1099
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_131
; %bb.130:
	s_movk_i32 s22, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s22 offen
.LBB3_131:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_or_b32_e32 v9, 0x70, v34
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[56:57], -1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[22:23], s40, v9
	s_cbranch_vccnz .LBB3_135
; %bb.132:
	s_and_b64 s[56:57], s[22:23], s[30:31]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB3_134
; %bb.133:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v8, off offset:448
.LBB3_134:                              ; %Flow1096
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[56:57], 0
.LBB3_135:                              ; %Flow1097
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b64 vcc, exec, s[56:57]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_137
; %bb.136:
	s_movk_i32 s30, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s30 offen
.LBB3_137:                              ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v105, v106 offset1:20
	ds_store_2addr_b32 v75, v104, v103 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_141
; %bb.138:
	s_and_b64 s[56:57], s[16:17], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB3_140
; %bb.139:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:64
.LBB3_140:                              ; %Flow1094
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_141:                              ; %Flow1095
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_143
; %bb.142:
	s_add_co_i32 s30, s41, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_143:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_147
; %bb.144:
	s_and_b64 s[56:57], s[18:19], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB3_146
; %bb.145:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:192
.LBB3_146:                              ; %Flow1092
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_147:                              ; %Flow1093
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_149
; %bb.148:
	s_add_co_i32 s30, s41, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_149:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_153
; %bb.150:
	s_and_b64 s[56:57], s[20:21], s[34:35]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_saveexec_b64 s[30:31], s[56:57]
	s_cbranch_execz .LBB3_152
; %bb.151:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:320
.LBB3_152:                              ; %Flow1090
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_153:                              ; %Flow1091
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_155
; %bb.154:
	s_add_co_i32 s30, s41, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_155:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_159
; %bb.156:
	s_and_b64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_158
; %bb.157:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_158:                              ; %Flow1088
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_159:                              ; %Flow1089
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_161
; %bb.160:
	s_addk_co_i32 s41, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s41 offen
.LBB3_161:                              ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v101, v102 offset1:20
	ds_store_2addr_b32 v75, v99, v100 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_165
; %bb.162:
	s_and_b64 s[34:35], s[16:17], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_164
; %bb.163:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB3_164:                              ; %Flow1086
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_165:                              ; %Flow1087
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_167
; %bb.166:
	s_or_b32 s30, s58, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_167:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_171
; %bb.168:
	s_and_b64 s[34:35], s[18:19], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_170
; %bb.169:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB3_170:                              ; %Flow1084
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_171:                              ; %Flow1085
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_173
; %bb.172:
	s_add_co_i32 s30, s58, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_173:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_177
; %bb.174:
	s_and_b64 s[34:35], s[20:21], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_176
; %bb.175:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB3_176:                              ; %Flow1082
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_177:                              ; %Flow1083
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_179
; %bb.178:
	s_add_co_i32 s30, s58, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_179:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_183
; %bb.180:
	s_and_b64 s[34:35], s[22:23], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_182
; %bb.181:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_182:                              ; %Flow1080
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_183:                              ; %Flow1081
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_185
; %bb.184:
	s_addk_co_i32 s58, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s58 offen
.LBB3_185:                              ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v95, v98 offset1:20
	ds_store_2addr_b32 v75, v96, v97 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_189
; %bb.186:
	s_and_b64 s[34:35], s[16:17], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_188
; %bb.187:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB3_188:                              ; %Flow1078
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_189:                              ; %Flow1079
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_191
; %bb.190:
	s_add_co_i32 s30, s59, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_191:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_195
; %bb.192:
	s_and_b64 s[34:35], s[18:19], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_194
; %bb.193:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB3_194:                              ; %Flow1076
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_195:                              ; %Flow1077
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_197
; %bb.196:
	s_add_co_i32 s30, s59, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_197:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_201
; %bb.198:
	s_and_b64 s[34:35], s[20:21], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_200
; %bb.199:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB3_200:                              ; %Flow1074
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_201:                              ; %Flow1075
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_203
; %bb.202:
	s_add_co_i32 s30, s59, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s30 offen
.LBB3_203:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[30:31], -1
	s_cbranch_vccnz .LBB3_207
; %bb.204:
	s_and_b64 s[34:35], s[22:23], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[30:31], s[34:35]
	s_cbranch_execz .LBB3_206
; %bb.205:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_206:                              ; %Flow1072
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	s_mov_b64 s[30:31], 0
.LBB3_207:                              ; %Flow1073
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_209
; %bb.208:
	s_addk_co_i32 s59, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s59 offen
.LBB3_209:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_tileILb0ELi2EEvPKcPK12block_i4_128Pfiiiii.exit.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b64 s[30:31], s[24:25]
	s_cbranch_execz .LBB3_213
; %bb.210:                              ; %.lr.ph.i.1.i
	v_mov_b32_e32 v0, 0
	s_and_saveexec_b64 s[24:25], s[28:29]
	s_cbranch_execz .LBB3_212
; %bb.211:
	global_load_b32 v0, v[44:45], off
.LBB3_212:                              ; %.preheader506.loopexit.i.1.i
	s_or_b64 exec, exec, s[24:25]
	global_load_b32 v1, v[42:43], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v86, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cndmask_b32_e64 v1, 0, v1, s[26:27]
	ds_store_2addr_stride64_b32 v85, v0, v1 offset0:48 offset1:56
.LBB3_213:                              ; %Flow1071
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[30:31]
	global_load_b64 v[0:1], v[50:51], off offset:8
	global_load_b64 v[2:3], v[52:53], off offset:8
	global_load_b64 v[4:5], v[46:47], off offset:8
	global_load_b64 v[6:7], v[48:49], off offset:8
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v77, 0
	s_mov_b32 s25, 0
	s_and_not1_b64 vcc, exec, s[54:55]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v0, 0, v0, s[4:5]
	v_cndmask_b32_e64 v1, 0, v1, s[4:5]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v4, 0, v4, s[6:7]
	v_cndmask_b32_e64 v5, 0, v5, s[6:7]
	ds_store_b32 v91, v2 offset:4096
	ds_store_2addr_b32 v91, v0, v1 offset1:16
	ds_store_b32 v91, v3 offset:4160
	s_wait_loadcnt 0x0
	ds_store_b32 v93, v6 offset:4096
	ds_store_2addr_b32 v93, v4, v5 offset1:16
	ds_store_b32 v93, v7 offset:4160
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_222
; %bb.214:                              ; %.preheader505.lr.ph.i.1.i
	v_or_b32_e32 v94, 0xf00, v94
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	s_movk_i32 s38, 0x1000
	s_movk_i32 s41, 0x3000
	s_movk_i32 s54, 0x3800
	s_mov_b32 s39, 0
	s_mov_b32 s26, s25
	s_branch .LBB3_216
.LBB3_215:                              ;   in Loop: Header=BB3_216 Depth=1
	s_and_not1_b64 vcc, exec, s[28:29]
	s_mov_b32 s26, s55
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_222
.LBB3_216:                              ; %.preheader505.i.1.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_218 Depth 2
	s_mov_b32 s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s55, s26, 1
	s_mul_u64 s[30:31], s[26:27], 0x88
	s_lshl_b32 s27, s26, 1
	s_cmp_eq_u32 s55, s33
	s_mov_b32 s56, 0
	s_cselect_b64 s[28:29], -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[48:49], s[30:31]
	s_mov_b64 s[36:37], 0
	s_mov_b64 s[34:35], -1
	s_branch .LBB3_218
.LBB3_217:                              ;   in Loop: Header=BB3_218 Depth=2
	s_wait_loadcnt_dscnt 0x307
	v_mul_f32_e32 v57, v71, v55
	v_cvt_f32_i32_e32 v28, v28
	v_mul_f32_e32 v58, v69, v55
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v56, v56
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v59, v70, v55
	v_mul_f32_e32 v28, v57, v28
	v_mul_f32_e32 v57, v72, v55
	v_mul_f32_e32 v29, v58, v29
	v_mul_f32_e32 v58, v65, v55
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v28, v57, v56
	v_fmac_f32_e32 v29, v59, v56
	v_mul_f32_e32 v57, v67, v55
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v25, v25
	v_add_f32_e32 v119, v119, v28
	v_add_f32_e32 v118, v118, v29
	v_mul_f32_e32 v28, v58, v30
	v_mul_f32_e32 v29, v66, v55
	v_mul_f32_e32 v30, v57, v31
	s_wait_dscnt 0x6
	v_mul_f32_e32 v57, v71, v53
	v_mul_f32_e32 v31, v68, v55
	v_cvt_f32_i32_e32 v26, v26
	v_fmac_f32_e32 v28, v29, v56
	v_cvt_f32_i32_e32 v29, v54
	v_mul_f32_e32 v54, v69, v53
	v_mul_f32_e32 v24, v57, v24
	v_mul_f32_e32 v57, v72, v53
	v_fmac_f32_e32 v30, v31, v56
	v_add_f32_e32 v117, v117, v28
	v_mul_f32_e32 v25, v54, v25
	v_mul_f32_e32 v28, v70, v53
	v_fmac_f32_e32 v24, v57, v29
	v_mul_f32_e32 v31, v65, v53
	v_mul_f32_e32 v54, v67, v53
	v_cvt_f32_i32_e32 v27, v27
	v_add_f32_e32 v116, v116, v30
	v_fmac_f32_e32 v25, v28, v29
	v_add_f32_e32 v115, v115, v24
	v_mul_f32_e32 v24, v31, v26
	v_mul_f32_e32 v26, v54, v27
	v_mul_f32_e32 v27, v66, v53
	v_mul_f32_e32 v28, v68, v53
	s_wait_dscnt 0x5
	v_mul_f32_e32 v30, v71, v51
	v_mul_f32_e32 v31, v69, v51
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_fmac_f32_e32 v24, v27, v29
	v_fmac_f32_e32 v26, v28, v29
	v_cvt_f32_i32_e32 v27, v52
	v_mul_f32_e32 v20, v30, v20
	v_mul_f32_e32 v21, v31, v21
	v_mul_f32_e32 v28, v72, v51
	v_mul_f32_e32 v30, v70, v51
	v_add_f32_e32 v113, v113, v24
	v_mul_f32_e32 v24, v65, v51
	v_cvt_f32_i32_e32 v22, v22
	v_fmac_f32_e32 v20, v28, v27
	v_fmac_f32_e32 v21, v30, v27
	v_add_f32_e32 v114, v114, v25
	v_mul_f32_e32 v25, v67, v51
	v_cvt_f32_i32_e32 v23, v23
	v_add_f32_e32 v111, v111, v20
	v_add_f32_e32 v110, v110, v21
	v_mul_f32_e32 v20, v24, v22
	v_mul_f32_e32 v21, v66, v51
	s_wait_dscnt 0x4
	v_mul_f32_e32 v24, v71, v41
	v_cvt_f32_i32_e32 v16, v16
	v_mul_f32_e32 v22, v25, v23
	v_mul_f32_e32 v23, v68, v51
	v_fmac_f32_e32 v20, v21, v27
	v_cvt_f32_i32_e32 v21, v42
	v_mul_f32_e32 v25, v69, v41
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v16, v24, v16
	v_mul_f32_e32 v24, v72, v41
	v_fmac_f32_e32 v22, v23, v27
	v_add_f32_e32 v109, v109, v20
	v_mul_f32_e32 v17, v25, v17
	v_mul_f32_e32 v20, v70, v41
	v_fmac_f32_e32 v16, v24, v21
	v_mul_f32_e32 v23, v65, v41
	v_mul_f32_e32 v24, v67, v41
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_add_f32_e32 v108, v108, v22
	v_fmac_f32_e32 v17, v20, v21
	v_add_f32_e32 v107, v107, v16
	v_mul_f32_e32 v16, v23, v18
	v_mul_f32_e32 v18, v24, v19
	v_mul_f32_e32 v19, v66, v41
	s_wait_dscnt 0x3
	v_mul_f32_e32 v22, v55, v47
	s_wait_dscnt 0x2
	v_mul_f32_e32 v23, v55, v49
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_add_f32_e32 v106, v106, v17
	v_fmac_f32_e32 v16, v19, v21
	v_mul_f32_e32 v17, v55, v48
	v_mul_f32_e32 v12, v22, v12
	v_mul_f32_e32 v13, v23, v13
	v_mul_f32_e32 v19, v55, v50
	v_mul_f32_e32 v20, v68, v41
	v_cvt_f32_i32_e32 v14, v14
	v_fmac_f32_e32 v12, v17, v56
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v13, v19, v56
	v_fmac_f32_e32 v18, v20, v21
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v55, v45
	v_add_f32_e32 v103, v103, v12
	v_mul_f32_e32 v12, v53, v47
	v_add_f32_e32 v102, v102, v13
	v_mul_f32_e32 v13, v53, v49
	v_cvt_f32_i32_e32 v9, v9
	v_add_f32_e32 v105, v105, v16
	v_mul_f32_e32 v14, v20, v14
	v_mul_f32_e32 v16, v55, v46
	v_mul_f32_e32 v8, v12, v8
	v_mul_f32_e32 v12, v53, v48
	v_mul_f32_e32 v9, v13, v9
	v_mul_f32_e32 v13, v53, v45
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v14, v16, v56
	v_fmac_f32_e32 v8, v12, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v12, v53, v43
	v_cvt_f32_i32_e32 v11, v11
	v_mul_f32_e32 v10, v13, v10
	v_mul_f32_e32 v13, v53, v46
	v_add_f32_e32 v101, v101, v14
	v_mul_f32_e32 v14, v53, v50
	v_add_f32_e32 v99, v99, v8
	v_mul_f32_e32 v8, v12, v11
	v_mul_f32_e32 v11, v53, v44
	v_fmac_f32_e32 v10, v13, v29
	v_mul_f32_e32 v12, v51, v47
	v_cvt_f32_i32_e32 v4, v4
	v_fmac_f32_e32 v9, v14, v29
	v_fmac_f32_e32 v8, v11, v29
	v_add_f32_e32 v97, v97, v10
	v_mul_f32_e32 v10, v51, v48
	v_mul_f32_e32 v4, v12, v4
	v_mul_f32_e32 v11, v51, v45
	v_mul_f32_e32 v12, v51, v43
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_add_f32_e32 v98, v98, v9
	v_mul_f32_e32 v9, v51, v49
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v4, v10, v27
	v_mul_f32_e32 v6, v11, v6
	v_mul_f32_e32 v7, v12, v7
	v_mul_f32_e32 v10, v51, v46
	v_mul_f32_e32 v11, v51, v44
	v_mul_f32_e32 v5, v9, v5
	v_mul_f32_e32 v9, v51, v50
	v_add_f32_e32 v95, v95, v4
	v_fmac_f32_e32 v6, v10, v27
	v_fmac_f32_e32 v7, v11, v27
	v_mul_f32_e32 v4, v41, v47
	v_cvt_f32_i32_e32 v0, v0
	v_mul_f32_e32 v22, v55, v43
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v96, v96, v8
	v_fmac_f32_e32 v5, v9, v27
	v_mul_f32_e32 v8, v41, v49
	v_cvt_f32_i32_e32 v1, v1
	v_add_f32_e32 v82, v82, v6
	v_add_f32_e32 v81, v81, v7
	v_mul_f32_e32 v0, v4, v0
	v_mul_f32_e32 v4, v41, v48
	v_mul_f32_e32 v6, v41, v45
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v7, v41, v43
	v_cvt_f32_i32_e32 v3, v3
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v15, v22, v15
	v_mul_f32_e32 v17, v55, v44
	v_add_f32_e32 v83, v83, v5
	v_mul_f32_e32 v1, v8, v1
	v_mul_f32_e32 v5, v41, v50
	v_mul_f32_e32 v2, v6, v2
	v_mul_f32_e32 v6, v41, v46
	v_fmac_f32_e32 v0, v4, v21
	v_mul_f32_e32 v3, v7, v3
	v_mul_f32_e32 v4, v41, v44
	v_fmac_f32_e32 v15, v17, v56
	v_fmac_f32_e32 v1, v5, v21
	v_fmac_f32_e32 v2, v6, v21
	v_add_f32_e32 v112, v112, v26
	v_fmac_f32_e32 v3, v4, v21
	v_add_f32_e32 v104, v104, v18
	v_add_f32_e32 v100, v100, v15
	v_add_f32_e32 v77, v77, v0
	v_add_f32_e32 v80, v80, v1
	v_add_f32_e32 v79, v79, v2
	v_add_f32_e32 v78, v78, v3
	s_xor_b64 s[36:37], s[34:35], -1
	s_mov_b32 s56, 1
	s_mov_b64 s[34:35], 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[36:37]
	s_mov_b64 s[36:37], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_215
.LBB3_218:                              ;   Parent Loop BB3_216 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s24, s56, s27
	s_lshl_b32 s58, s56, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[24:25], s[42:43]
	s_mov_b32 s59, s25
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[60:61], s[60:61], 0x48
	s_add_nc_u64 s[58:59], s[30:31], s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[60:61], s[50:51], s[60:61]
	v_add_nc_u32_e32 v45, s38, v90
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, s[62:63], s60, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s61, 0, s[62:63]
	v_add_co_u32 v2, s[62:63], s58, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s59, 0, s[62:63]
	v_add_co_u32 v4, s[62:63], s60, v38
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s61, 0, s[62:63]
	v_add_co_u32 v6, s[60:61], s58, v39
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s59, 0, s[60:61]
	global_load_b64 v[63:64], v[0:1], off offset:40
	global_load_b64 v[57:58], v[2:3], off offset:40
	global_load_b64 v[61:62], v[4:5], off offset:40
	global_load_b64 v[59:60], v[6:7], off offset:40
	v_add_nc_u32_e32 v46, s39, v87
	ds_load_2addr_stride64_b32 v[41:42], v45 offset1:2
	ds_load_2addr_stride64_b32 v[0:1], v46 offset0:8 offset1:10
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:12 offset1:14
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v1, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v44, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v0, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v1, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v43, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v44, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[41:42], v45 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[43:44], v46 offset0:9 offset1:11
	v_add_nc_u32_e32 v45, s39, v94
	ds_load_b32 v46, v46 offset:3328
	ds_load_b32 v45, v45
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v43, 0x2000, v91
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v63, s[4:5]
	v_add_nc_u32_e32 v42, 0x4000, v91
	v_cndmask_b32_e64 v44, 0, v64, s[4:5]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v43, v57, v58 offset1:16
	ds_store_2addr_b32 v42, v41, v44 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v61, s[6:7]
	v_cndmask_b32_e64 v42, 0, v62, s[6:7]
	v_add_nc_u32_e32 v43, 0x4000, v93
	v_add_nc_u32_e32 v44, 0x2000, v93
	ds_store_2addr_b32 v43, v41, v42 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v44, v59, v60 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[38:39], s[36:37], s[28:29]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[38:39]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_220
; %bb.219:                              ;   in Loop: Header=BB3_218 Depth=2
	s_add_co_i32 s24, s24, 1
	s_xor_b32 s64, s56, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[24:25], s[42:43]
	s_add_co_i32 s24, s56, s26
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[58:59], s[58:59], 0x48
	s_mul_u64 s[60:61], s[24:25], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[58:59], s[50:51], s[58:59]
	s_add_nc_u64 s[56:57], s[48:49], s[60:61]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[45:46], null, 0x48, v88, s[58:59]
	v_add_co_u32 v41, s[62:63], s58, v36
	s_lshl_b32 s60, s64, 6
	s_mov_b32 s61, s25
	s_mulk_i32 s24, 0x88
	v_add_co_ci_u32_e64 v42, null, s59, 0, s[62:63]
	v_add_co_u32 v45, vcc, v45, v84
	v_add_co_u32 v43, s[62:63], s58, v38
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[56:57], s[56:57], s[60:61]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v46, null, 0, v46, vcc
	v_add_co_u32 v51, vcc, v33, s24
	v_add_co_ci_u32_e64 v44, null, s59, 0, s[62:63]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v47, s[58:59], s56, v37
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v40, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v48, null, s57, 0, s[58:59]
	v_add_co_u32 v49, s[58:59], s56, v39
	s_lshl_b32 s24, s64, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v50, null, s57, 0, s[58:59]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v51, vcc, v51, s24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v52, null, 0, v52, vcc
	s_clause 0x1
	global_load_b64 v[63:64], v[41:42], off offset:8
	global_load_b64 v[61:62], v[43:44], off offset:8
	s_clause 0x1
	global_load_b64 v[57:58], v[47:48], off offset:8
	global_load_b64 v[59:60], v[49:50], off offset:8
	global_load_b32 v120, v[45:46], off
	global_load_b32 v121, v[51:52], off
.LBB3_220:                              ; %.preheader503.i.1.i
                                        ;   in Loop: Header=BB3_218 Depth=2
	v_add_nc_u32_e32 v47, 0, v90
	v_add_nc_u32_e32 v48, 0, v87
	s_xor_b64 s[38:39], s[38:39], -1
	s_and_b64 s[56:57], s[34:35], exec
	s_cselect_b32 s24, s41, 0x3400
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:72 offset1:74
	ds_load_2addr_stride64_b32 v[45:46], v48 offset0:76 offset1:78
	s_cselect_b32 s56, s54, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v45, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v46, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v45, v[4:7] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v46, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v45, 0, v94
	ds_load_2addr_stride64_b32 v[41:42], v47 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[43:44], v48 offset0:73 offset1:75
	ds_load_b32 v46, v48 offset:19712
	ds_load_b32 v45, v45 offset:16384
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[28:31], v41, v43, v[28:31] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[24:27], v41, v44, v[24:27] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[20:23], v41, v46, v[20:23] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[12:15], v42, v43, v[12:15] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[8:11], v42, v44, v[8:11] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[4:7], v42, v46, v[4:7] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[16:19], v41, v45, v[16:19] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[0:3], v42, v45, v[0:3] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v43, s56, v92
	v_add_nc_u32_e32 v41, s24, v89
	s_and_not1_b64 vcc, exec, s[38:39]
	s_movk_i32 s38, 0x2000
	s_movk_i32 s39, 0x4000
	ds_load_2addr_b32 v[71:72], v43 offset1:1
	ds_load_2addr_b32 v[69:70], v43 offset0:2 offset1:3
	ds_load_2addr_b32 v[65:66], v43 offset0:4 offset1:5
	ds_load_2addr_b32 v[67:68], v43 offset0:6 offset1:7
	ds_load_2addr_b32 v[55:56], v41 offset0:128 offset1:129
	ds_load_2addr_b32 v[53:54], v41 offset0:160 offset1:161
	ds_load_2addr_b32 v[51:52], v41 offset0:192 offset1:193
	ds_load_2addr_b32 v[41:42], v41 offset0:224 offset1:225
	ds_load_2addr_b32 v[47:48], v43 offset0:32 offset1:33
	ds_load_2addr_b32 v[49:50], v43 offset0:34 offset1:35
	ds_load_2addr_b32 v[45:46], v43 offset0:36 offset1:37
	ds_load_2addr_b32 v[43:44], v43 offset0:38 offset1:39
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_217
; %bb.221:                              ; %.preheader504.i.1.i
                                        ;   in Loop: Header=BB3_218 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v122, v86, v121
	s_and_b64 s[36:37], s[36:37], exec
	v_cndmask_b32_e64 v63, 0, v63, s[4:5]
	v_cndmask_b32_e64 v64, 0, v64, s[4:5]
	s_cselect_b32 s24, s54, 0x3c00
	v_cvt_f32_f16_e32 v122, v122.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v123, s24, v85
	ds_store_b32 v91, v57 offset:4096
	ds_store_2addr_b32 v91, v63, v64 offset1:16
	v_cndmask_b32_e64 v57, 0, v61, s[6:7]
	v_cndmask_b32_e64 v61, 0, v62, s[6:7]
	v_cndmask_b32_e64 v122, 0, v122, s[2:3]
	s_cselect_b32 s24, s41, 0x3400
	s_mov_b32 s39, 0
	s_movk_i32 s38, 0x1000
	v_cndmask_b32_e64 v62, 0, v120, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v63, s24, v85
	ds_store_b32 v91, v58 offset:4160
	ds_store_b32 v93, v59 offset:4096
	ds_store_2addr_b32 v93, v57, v61 offset1:16
	ds_store_b32 v93, v60 offset:4160
	ds_store_b32 v63, v62
	ds_store_b32 v123, v122
	s_branch .LBB3_217
.LBB3_222:                              ; %._crit_edge575.i.1.i
	ds_store_2addr_b32 v75, v119, v118 offset1:20
	ds_store_2addr_b32 v75, v117, v116 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v3, 64, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[0:1], null, s40, v3, 0
	v_cmp_gt_i32_e64 s[0:1], s42, v3
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v2, v32
	v_add_co_u32 v0, s[2:3], s52, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v1, null, s53, v1, s[2:3]
	s_mov_b64 s[2:3], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_226
; %bb.223:
	s_and_b64 s[4:5], s[8:9], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB3_225
; %bb.224:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off
.LBB3_225:                              ; %Flow1065
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB3_226:                              ; %Flow1066
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_lshl_b32 s26, s40, 8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_228
; %bb.227:
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v73, s[44:47], s26 offen
.LBB3_228:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_232
; %bb.229:
	s_and_b64 s[4:5], s[10:11], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB3_231
; %bb.230:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:128
.LBB3_231:                              ; %Flow1063
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB3_232:                              ; %Flow1064
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_234
; %bb.233:
	s_or_b32 s2, s26, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB3_234:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_238
; %bb.235:
	s_and_b64 s[4:5], s[12:13], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB3_237
; %bb.236:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:256
.LBB3_237:                              ; %Flow1061
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB3_238:                              ; %Flow1062
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_240
; %bb.239:
	s_add_co_i32 s2, s26, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB3_240:
	s_wait_dscnt 0x0
	ds_load_b32 v2, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_244
; %bb.241:
	s_and_b64 s[4:5], s[14:15], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB3_243
; %bb.242:
	v_lshlrev_b64_e32 v[3:4], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v1, v4, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v2, off offset:384
.LBB3_243:                              ; %Flow1059
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	s_mov_b64 s[2:3], 0
.LBB3_244:                              ; %Flow1060
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_246
; %bb.245:
	s_add_co_i32 s2, s26, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v73, s[44:47], s2 offen
.LBB3_246:                              ; %.preheader.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, 0x50, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[2:3], null, s40, v5, 0
	v_cmp_gt_i32_e64 s[2:3], s42, v5
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[2:3], 2, v[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v115, v114 offset1:20
	ds_store_2addr_b32 v75, v113, v112 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v2, s[4:5], s52, v2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v3, null, s53, v3, s[4:5]
	s_mov_b64 s[4:5], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_250
; %bb.247:
	s_and_b64 s[6:7], s[8:9], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB3_249
; %bb.248:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off
.LBB3_249:                              ; %Flow1057
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB3_250:                              ; %Flow1058
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_mul_i32 s27, s40, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_252
; %bb.251:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s27 offen
.LBB3_252:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB3_256
; %bb.253:
	s_and_b64 s[6:7], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB3_255
; %bb.254:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:128
.LBB3_255:                              ; %Flow1055
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB3_256:                              ; %Flow1056
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_258
; %bb.257:
	s_add_co_i32 s4, s27, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB3_258:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB3_262
; %bb.259:
	s_and_b64 s[6:7], s[12:13], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB3_261
; %bb.260:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:256
.LBB3_261:                              ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB3_262:                              ; %Flow1054
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_264
; %bb.263:
	s_add_co_i32 s4, s27, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB3_264:
	s_wait_dscnt 0x0
	ds_load_b32 v4, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[4:5], -1
	s_cbranch_vccnz .LBB3_268
; %bb.265:
	s_and_b64 s[6:7], s[14:15], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB3_267
; %bb.266:
	v_lshlrev_b64_e32 v[5:6], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v3, v6, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v4, off offset:384
.LBB3_267:                              ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[4:5]
	s_mov_b64 s[4:5], 0
.LBB3_268:                              ; %Flow1052
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_270
; %bb.269:
	s_add_co_i32 s4, s27, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v73, s[44:47], s4 offen
.LBB3_270:                              ; %.preheader.2.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v7, 0x60, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[4:5], null, s40, v7, 0
	v_cmp_gt_i32_e64 s[4:5], s42, v7
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v111, v110 offset1:20
	ds_store_2addr_b32 v75, v109, v108 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v4, s[6:7], s52, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s53, v5, s[6:7]
	s_mov_b64 s[6:7], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_274
; %bb.271:
	s_and_b64 s[24:25], s[8:9], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB3_273
; %bb.272:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off
.LBB3_273:                              ; %Flow1049
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB3_274:                              ; %Flow1050
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_mul_i32 s28, s40, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_276
; %bb.275:
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s28 offen
.LBB3_276:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB3_280
; %bb.277:
	s_and_b64 s[24:25], s[10:11], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB3_279
; %bb.278:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:128
.LBB3_279:                              ; %Flow1047
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB3_280:                              ; %Flow1048
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_282
; %bb.281:
	s_add_co_i32 s6, s28, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB3_282:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB3_286
; %bb.283:
	s_and_b64 s[24:25], s[12:13], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB3_285
; %bb.284:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:256
.LBB3_285:                              ; %Flow1045
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB3_286:                              ; %Flow1046
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_288
; %bb.287:
	s_add_co_i32 s6, s28, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB3_288:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[6:7], -1
	s_cbranch_vccnz .LBB3_292
; %bb.289:
	s_and_b64 s[24:25], s[14:15], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB3_291
; %bb.290:
	v_lshlrev_b64_e32 v[7:8], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v4, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v5, v8, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:384
.LBB3_291:                              ; %Flow1043
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[6:7]
	s_mov_b64 s[6:7], 0
.LBB3_292:                              ; %Flow1044
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_294
; %bb.293:
	s_add_co_i32 s6, s28, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v73, s[44:47], s6 offen
.LBB3_294:                              ; %.preheader.3.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 0x70, v76
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_i64_i32 v[6:7], null, s40, v9, 0
	v_cmp_gt_i32_e64 s[6:7], s42, v9
	s_and_b64 vcc, exec, vcc
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v107, v106 offset1:20
	ds_store_2addr_b32 v75, v105, v104 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v6, s[24:25], s52, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s53, v7, s[24:25]
	s_mov_b64 s[24:25], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_298
; %bb.295:
	s_and_b64 s[24:25], s[8:9], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[24:25]
	s_cbranch_execz .LBB3_297
; %bb.296:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB3_297:                              ; %Flow1041
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[24:25], 0
.LBB3_298:                              ; %Flow1042
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[24:25]
	s_mul_i32 s24, s40, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_300
; %bb.299:
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s24 offen
.LBB3_300:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_304
; %bb.301:
	s_and_b64 s[10:11], s[10:11], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_303
; %bb.302:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:128
.LBB3_303:                              ; %Flow1039
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_304:                              ; %Flow1040
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_306
; %bb.305:
	s_add_co_i32 s8, s24, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_306:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_310
; %bb.307:
	s_and_b64 s[10:11], s[12:13], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_309
; %bb.308:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:256
.LBB3_309:                              ; %Flow1037
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_310:                              ; %Flow1038
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_312
; %bb.311:
	s_add_co_i32 s8, s24, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_312:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_316
; %bb.313:
	s_and_b64 s[10:11], s[14:15], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_315
; %bb.314:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v6, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v7, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:384
.LBB3_315:                              ; %Flow1035
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_316:                              ; %Flow1036
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_318
; %bb.317:
	s_add_co_i32 s8, s24, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_318:                              ; %.preheader500.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v103, v102 offset1:20
	ds_store_2addr_b32 v75, v101, v100 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v8, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_322
; %bb.319:
	s_and_b64 s[10:11], s[16:17], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_321
; %bb.320:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:64
.LBB3_321:                              ; %Flow1033
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_322:                              ; %Flow1034
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_324
; %bb.323:
	s_or_b32 s8, s26, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_324:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_328
; %bb.325:
	s_and_b64 s[10:11], s[18:19], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_327
; %bb.326:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:192
.LBB3_327:                              ; %Flow1031
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_328:                              ; %Flow1032
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_330
; %bb.329:
	s_or_b32 s8, s26, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_330:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_334
; %bb.331:
	s_and_b64 s[10:11], s[20:21], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB3_333
; %bb.332:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off offset:320
.LBB3_333:                              ; %Flow1029
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[8:9]
	s_mov_b64 s[8:9], 0
.LBB3_334:                              ; %Flow1030
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_336
; %bb.335:
	s_add_co_i32 s8, s26, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s8 offen
.LBB3_336:
	s_wait_dscnt 0x0
	ds_load_b32 v8, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[8:9], -1
	s_cbranch_vccnz .LBB3_340
; %bb.337:
	s_and_b64 s[8:9], s[22:23], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_339
; %bb.338:
	v_lshlrev_b64_e32 v[9:10], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v0, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v8, off offset:448
.LBB3_339:                              ; %Flow1027
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[8:9], 0
.LBB3_340:                              ; %Flow1028
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[8:9]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_342
; %bb.341:
	s_addk_co_i32 s26, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v73, s[44:47], s26 offen
.LBB3_342:                              ; %.preheader.1.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v99, v98 offset1:20
	ds_store_2addr_b32 v75, v97, v96 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_346
; %bb.343:
	s_and_b64 s[8:9], s[16:17], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_345
; %bb.344:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:64
.LBB3_345:                              ; %Flow1025
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_346:                              ; %Flow1026
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_348
; %bb.347:
	s_add_co_i32 s0, s27, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_348:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_352
; %bb.349:
	s_and_b64 s[8:9], s[18:19], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_351
; %bb.350:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:192
.LBB3_351:                              ; %Flow1023
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_352:                              ; %Flow1024
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_354
; %bb.353:
	s_add_co_i32 s0, s27, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_354:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_358
; %bb.355:
	s_and_b64 s[8:9], s[20:21], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB3_357
; %bb.356:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v0, off offset:320
.LBB3_357:                              ; %Flow1021
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_358:                              ; %Flow1022
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_360
; %bb.359:
	s_add_co_i32 s0, s27, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_360:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_364
; %bb.361:
	s_and_b64 s[2:3], s[22:23], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_363
; %bb.362:
	v_lshlrev_b64_e32 v[8:9], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v2, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v3, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_363:                              ; %Flow1019
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_364:                              ; %Flow1020
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_366
; %bb.365:
	s_addk_co_i32 s27, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s27 offen
.LBB3_366:                              ; %.preheader.2.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v95, v83 offset1:20
	ds_store_2addr_b32 v75, v82, v81 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_370
; %bb.367:
	s_and_b64 s[2:3], s[16:17], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_369
; %bb.368:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB3_369:                              ; %Flow1017
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_370:                              ; %Flow1018
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_372
; %bb.371:
	s_or_b32 s0, s28, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_372:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_376
; %bb.373:
	s_and_b64 s[2:3], s[18:19], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_375
; %bb.374:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB3_375:                              ; %Flow1015
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_376:                              ; %Flow1016
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_378
; %bb.377:
	s_add_co_i32 s0, s28, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_378:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_382
; %bb.379:
	s_and_b64 s[2:3], s[20:21], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_381
; %bb.380:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB3_381:                              ; %Flow1013
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_382:                              ; %Flow1014
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_384
; %bb.383:
	s_add_co_i32 s0, s28, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_384:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_388
; %bb.385:
	s_and_b64 s[2:3], s[22:23], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_387
; %bb.386:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_387:                              ; %Flow1011
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_388:                              ; %Flow1012
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_390
; %bb.389:
	s_addk_co_i32 s28, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s28 offen
.LBB3_390:                              ; %.preheader.3.1.i.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v75, v77, v80 offset1:20
	ds_store_2addr_b32 v75, v79, v78 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v0, v32
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_394
; %bb.391:
	s_and_b64 s[2:3], s[16:17], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_393
; %bb.392:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:64
.LBB3_393:                              ; %Flow1009
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_394:                              ; %Flow1010
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_396
; %bb.395:
	s_add_co_i32 s0, s24, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_396:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_400
; %bb.397:
	s_and_b64 s[2:3], s[18:19], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_399
; %bb.398:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:192
.LBB3_399:                              ; %Flow1007
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_400:                              ; %Flow1008
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_402
; %bb.401:
	s_add_co_i32 s0, s24, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_402:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_406
; %bb.403:
	s_and_b64 s[2:3], s[20:21], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_405
; %bb.404:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:320
.LBB3_405:                              ; %Flow1005
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_406:                              ; %Flow1006
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_408
; %bb.407:
	s_add_co_i32 s0, s24, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s0 offen
.LBB3_408:
	s_wait_dscnt 0x0
	ds_load_b32 v0, v32 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v74
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_412
; %bb.409:
	s_and_b64 s[2:3], s[22:23], s[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_411
; %bb.410:
	v_lshlrev_b64_e32 v[1:2], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v7, v2, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v0, off offset:448
.LBB3_411:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_412:                              ; %Flow1004
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_414
; %bb.413:
	s_addk_co_i32 s24, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v0, v73, s[44:47], s24 offen
.LBB3_414:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_endpgm
.Lfunc_end3:
	.size	iu4_w64_r2_set, .Lfunc_end3-iu4_w64_r2_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r2_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 132
		.amdhsa_next_free_sgpr 65
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-iu4_w64_r2_set)<<4)&4080)>>4
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
	.set .Liu4_w64_r2_set.num_vgpr, 132
	.set .Liu4_w64_r2_set.num_agpr, 0
	.set .Liu4_w64_r2_set.numbered_sgpr, 65
	.set .Liu4_w64_r2_set.num_named_barrier, 0
	.set .Liu4_w64_r2_set.private_seg_size, 0
	.set .Liu4_w64_r2_set.uses_vcc, 1
	.set .Liu4_w64_r2_set.uses_flat_scratch, 0
	.set .Liu4_w64_r2_set.has_dyn_sized_stack, 0
	.set .Liu4_w64_r2_set.has_recursion, 0
	.set .Liu4_w64_r2_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16876
; TotalNumSgprs: 67
; NumVgprs: 132
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 32
; NumSGPRsForWavesPerEU: 67
; NumVGPRsForWavesPerEU: 132
; Occupancy: 5
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
	.type	__hip_cuid_6ed54641408283bb,@object ; @__hip_cuid_6ed54641408283bb
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6ed54641408283bb
__hip_cuid_6ed54641408283bb:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6ed54641408283bb, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6ed54641408283bb
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           quantize_int4_mmq_ds128
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         quantize_int4_mmq_ds128.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           iu4_w64_r2
    .private_segment_fixed_size: 0
    .sgpr_count:     75
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     141
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           iu4_w64_r2_add
    .private_segment_fixed_size: 0
    .sgpr_count:     67
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r2_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     132
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           iu4_w64_r2_set
    .private_segment_fixed_size: 0
    .sgpr_count:     67
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r2_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     132
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
