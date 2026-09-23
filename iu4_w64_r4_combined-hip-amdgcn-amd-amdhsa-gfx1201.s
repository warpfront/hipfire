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
	.protected	iu4_w64_r4              ; -- Begin function iu4_w64_r4
	.globl	iu4_w64_r4
	.p2align	8
	.type	iu4_w64_r4,@function
iu4_w64_r4:                             ; @iu4_w64_r4
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[8:11], s[0:1], 0x18
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	v_lshrrev_b32_e32 v122, 1, v0
	v_lshrrev_b32_e32 v120, 6, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v121, 64, v122
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s20, s8
	s_cselect_b64 s[0:1], -1, 0
	s_cmp_lt_i32 s18, s10
	s_cselect_b64 s[2:3], -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b64 s[22:23], s[0:1], s[2:3]
	s_cmp_eq_u32 s11, 0
	s_cbranch_scc1 .LBB1_8
; %bb.1:
	s_mov_b64 s[24:25], 0
	s_and_b64 vcc, exec, s[22:23]
	s_mov_b64 s[0:1], 0
	s_cbranch_vccz .LBB1_9
; %bb.2:                                ; %.preheader538.i.i
	s_ashr_i32 s0, s9, 31
	s_add_co_i32 s7, s8, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s9, s0
	s_ashr_i32 s19, s0, 8
	s_mov_b64 s[0:1], exec
	s_mul_i32 s6, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.i.i
	v_and_b32_e32 v1, 1, v0
	v_or_b32_e32 v4, s18, v122
	v_mov_b32_e32 v3, 0
	s_mov_b64 s[2:3], exec
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 2, v1
	v_cmpx_gt_i32_e64 s10, v4
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[3:4], null, 0x48, v4, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc, v3, v2
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc
	global_load_b32 v3, v[3:4], off
.LBB1_5:                                ; %.preheader535.loopexit.i.i
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v6, s20, v122
	v_lshlrev_b32_e32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v4, s7, v6
	v_cmp_gt_i32_e32 vcc, s8, v6
	v_mad_co_i64_i32 v[4:5], null, s6, v4, s[12:13]
	global_load_b32 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v1, v4
	v_lshlrev_b32_e32 v4, 3, v122
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_add3_u32 v2, 0, v4, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0, v1, vcc
	ds_store_2addr_stride64_b32 v2, v3, v1 offset0:48 offset1:56
.LBB1_6:                                ; %Flow2277
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v1, 2, v0
	s_add_co_i32 s21, s10, -1
	v_lshrrev_b32_e32 v11, 5, v0
	v_lshlrev_b32_e32 v12, 7, v0
	v_and_b32_e32 v13, 60, v0
	v_add_nc_u32_e32 v9, s18, v1
	v_add_nc_u32_e32 v2, s20, v1
	v_and_b32_e32 v1, 3, v0
	v_bfe_u32 v14, v0, 1, 1
	v_or_b32_e32 v15, 8, v11
	v_add_nc_u32_e32 v10, 64, v9
	v_add_nc_u32_e32 v3, 64, v2
	v_lshlrev_b32_e32 v1, 3, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s21, v9
	v_min_i32_e32 v2, s7, v2
	v_min_i32_e32 v5, s21, v10
	v_min_i32_e32 v3, s7, v3
	v_and_or_b32 v12, 0x80, v12, v13
	v_and_or_b32 v11, v11, 6, v14
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s6, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s6, v3, v[1:2]
	v_and_or_b32 v13, v15, 14, v14
	v_lshl_or_b32 v11, v11, 8, v12
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	global_load_b64 v[1:2], v68, s[14:15] offset:8
	global_load_b64 v[3:4], v69, s[12:13] offset:8
	global_load_b64 v[5:6], v70, s[14:15] offset:8
	global_load_b64 v[7:8], v71, s[12:13] offset:8
	v_lshl_or_b32 v12, v13, 8, v12
	v_add_nc_u32_e32 v183, 0, v11
	v_cmp_gt_i32_e64 s[2:3], s10, v10
	s_cmp_gt_i32 s9, 0xff
	v_add_nc_u32_e32 v184, 0, v12
	v_add_nc_u32_e32 v9, 0x1000, v183
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v10, 0x1000, v184
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v1, 0, v1, s[0:1]
	v_cndmask_b32_e64 v2, 0, v2, s[0:1]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v5, 0, v5, s[2:3]
	v_cndmask_b32_e64 v6, 0, v6, s[2:3]
	ds_store_2addr_b32 v9, v3, v4 offset1:16
	ds_store_2addr_b32 v183, v1, v2 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v10, v7, v8 offset1:16
	ds_store_2addr_b32 v184, v5, v6 offset1:16
	v_bfe_u32 v1, v0, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v182, 2, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_10
; %bb.7:                                ; %._crit_edge.._crit_edge604_crit_edge.i.i
	v_lshlrev_b32_e32 v6, 2, v1
	s_mov_b64 s[4:5], 0
	s_branch .LBB1_11
.LBB1_8:
	s_mov_b64 s[24:25], -1
	s_mov_b64 s[0:1], 0
.LBB1_9:                                ; %Flow2414
	s_and_b64 vcc, exec, s[24:25]
	s_cbranch_vccnz .LBB1_406
	s_branch .LBB1_808
.LBB1_10:
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr6
.LBB1_11:                               ; %Flow2275
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v123, 0
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_21
; %bb.12:                               ; %.preheader534.lr.ph.i.i
	v_add_nc_u32_e32 v1, s20, v122
	v_and_b32_e32 v2, 63, v0
	v_lshlrev_b32_e32 v3, 5, v0
	v_and_b32_e32 v6, 1, v0
	v_lshlrev_b32_e32 v9, 3, v0
	v_min_i32_e32 v5, s7, v1
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_nc_u32_e32 v4, s18, v122
	v_or_b32_e32 v7, v182, v121
	v_lshlrev_b32_e32 v8, 3, v122
	v_mad_co_u64_u32 v[65:66], null, s6, v5, s[12:13]
	v_ashrrev_i32_e32 v5, 31, v5
	v_lshlrev_b32_e32 v10, 2, v6
	v_and_or_b32 v190, 0x800, v3, v2
	v_lshl_or_b32 v191, v121, 5, v2
	v_and_b32_e32 v2, 0x278, v9
	v_mov_b32_e32 v123, 0
	v_min_i32_e32 v189, s21, v4
	v_mad_co_u64_u32 v[66:67], null, s6, v5, v[66:67]
	v_cmp_gt_i32_e64 s[4:5], s10, v4
	v_add3_u32 v192, 0, v8, v10
	v_cmp_gt_i32_e64 s[6:7], s8, v1
	v_lshlrev_b32_e32 v193, 4, v6
	v_lshl_add_u32 v194, v7, 3, 0
	v_add_nc_u32_e32 v195, 0, v2
	v_lshlrev_b32_e32 v196, 2, v6
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	s_mov_b32 s27, 0
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s40, 0x1000
	s_movk_i32 s21, 0x3000
	s_movk_i32 s33, 0x3800
	s_mov_b32 s41, 0
	s_mov_b32 s28, s27
	s_branch .LBB1_14
.LBB1_13:                               ;   in Loop: Header=BB1_14 Depth=1
	s_and_b64 vcc, exec, s[30:31]
	s_mov_b32 s28, s42
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_20
.LBB1_14:                               ; %.preheader534.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_16 Depth 2
	s_mov_b32 s29, s27
	s_add_co_i32 s42, s28, 1
	s_mul_u64 s[34:35], s[28:29], 0x88
	s_lshl_b32 s29, s28, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s42, s19
	s_mov_b64 s[38:39], 0
	s_cselect_b64 s[30:31], -1, 0
	s_add_nc_u64 s[34:35], s[12:13], s[34:35]
	s_mov_b64 s[36:37], -1
	s_mov_b32 s43, s27
	s_branch .LBB1_16
.LBB1_15:                               ;   in Loop: Header=BB1_16 Depth=2
	s_wait_loadcnt_dscnt 0x20f
	v_mul_f32_e32 v104, v116, v86
	v_cvt_f32_i32_e32 v61, v61
	v_cvt_f32_i32_e32 v87, v87
	v_mul_f32_e32 v105, v118, v86
	v_cvt_f32_i32_e32 v62, v62
	v_mul_f32_e32 v106, v117, v86
	v_mul_f32_e32 v61, v104, v61
	v_mul_f32_e32 v104, v119, v86
	v_mul_f32_e32 v107, v114, v86
	v_mul_f32_e32 v62, v105, v62
	v_mul_f32_e32 v105, v112, v86
	v_fmac_f32_e32 v61, v106, v87
	v_cvt_f32_i32_e32 v63, v63
	v_cvt_f32_i32_e32 v64, v64
	v_fmac_f32_e32 v62, v104, v87
	v_mul_f32_e32 v104, v113, v86
	v_add_f32_e32 v187, v187, v61
	v_mul_f32_e32 v61, v107, v63
	v_mul_f32_e32 v63, v105, v64
	v_mul_f32_e32 v64, v115, v86
	s_wait_dscnt 0xe
	v_mul_f32_e32 v106, v118, v84
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v107, v57
	v_fmac_f32_e32 v63, v104, v87
	v_fmac_f32_e32 v61, v64, v87
	v_cvt_f32_i32_e32 v57, v85
	v_mul_f32_e32 v58, v106, v58
	v_mul_f32_e32 v104, v119, v84
	v_add_f32_e32 v188, v188, v62
	v_add_f32_e32 v185, v185, v61
	v_mul_f32_e32 v61, v114, v84
	v_cvt_f32_i32_e32 v59, v59
	v_fmac_f32_e32 v58, v104, v57
	v_mul_f32_e32 v62, v112, v84
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v53, v53
	v_add_f32_e32 v186, v186, v63
	v_add_f32_e32 v181, v181, v58
	v_mul_f32_e32 v58, v61, v59
	v_mul_f32_e32 v59, v115, v84
	v_mul_f32_e32 v60, v62, v60
	s_wait_dscnt 0xd
	v_mul_f32_e32 v62, v116, v82
	v_mul_f32_e32 v61, v113, v84
	v_mul_f32_e32 v63, v118, v82
	v_fmac_f32_e32 v58, v59, v57
	v_cvt_f32_i32_e32 v59, v83
	v_cvt_f32_i32_e32 v54, v54
	v_mul_f32_e32 v53, v62, v53
	v_mul_f32_e32 v62, v117, v82
	v_fmac_f32_e32 v60, v61, v57
	v_add_f32_e32 v178, v178, v58
	v_mul_f32_e32 v54, v63, v54
	v_mul_f32_e32 v58, v119, v82
	v_fmac_f32_e32 v53, v62, v59
	v_mul_f32_e32 v61, v114, v82
	v_mul_f32_e32 v62, v112, v82
	v_cvt_f32_i32_e32 v55, v55
	v_cvt_f32_i32_e32 v56, v56
	v_add_f32_e32 v179, v179, v60
	v_fmac_f32_e32 v54, v58, v59
	v_add_f32_e32 v176, v176, v53
	v_mul_f32_e32 v53, v61, v55
	v_mul_f32_e32 v55, v62, v56
	v_mul_f32_e32 v56, v115, v82
	v_mul_f32_e32 v58, v113, v82
	s_wait_dscnt 0xc
	v_mul_f32_e32 v60, v116, v72
	v_mul_f32_e32 v61, v118, v72
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_fmac_f32_e32 v53, v56, v59
	v_fmac_f32_e32 v55, v58, v59
	v_cvt_f32_i32_e32 v56, v73
	v_mul_f32_e32 v49, v60, v49
	v_mul_f32_e32 v50, v61, v50
	v_mul_f32_e32 v58, v117, v72
	v_mul_f32_e32 v60, v119, v72
	v_add_f32_e32 v177, v177, v54
	v_add_f32_e32 v174, v174, v53
	v_mul_f32_e32 v53, v114, v72
	v_fmac_f32_e32 v49, v58, v56
	v_fmac_f32_e32 v50, v60, v56
	v_cvt_f32_i32_e32 v51, v51
	v_mul_f32_e32 v54, v112, v72
	v_cvt_f32_i32_e32 v52, v52
	v_add_f32_e32 v172, v172, v49
	v_add_f32_e32 v173, v173, v50
	v_mul_f32_e32 v49, v53, v51
	v_mul_f32_e32 v50, v115, v72
	v_mul_f32_e32 v51, v54, v52
	s_wait_dscnt 0xb
	v_mul_f32_e32 v52, v86, v100
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v53, v113, v72
	v_fmac_f32_e32 v49, v50, v56
	s_wait_dscnt 0xa
	v_mul_f32_e32 v50, v86, v102
	v_cvt_f32_i32_e32 v46, v46
	v_mul_f32_e32 v45, v52, v45
	v_mul_f32_e32 v52, v86, v101
	v_fmac_f32_e32 v51, v53, v56
	v_add_f32_e32 v170, v170, v49
	v_mul_f32_e32 v46, v50, v46
	v_mul_f32_e32 v49, v86, v103
	v_fmac_f32_e32 v45, v52, v87
	s_wait_dscnt 0x9
	v_mul_f32_e32 v50, v86, v96
	s_wait_dscnt 0x8
	v_mul_f32_e32 v52, v86, v98
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_add_f32_e32 v171, v171, v51
	v_fmac_f32_e32 v46, v49, v87
	v_add_f32_e32 v168, v168, v45
	v_mul_f32_e32 v45, v50, v47
	v_mul_f32_e32 v47, v52, v48
	v_mul_f32_e32 v48, v86, v97
	v_mul_f32_e32 v50, v84, v100
	v_mul_f32_e32 v51, v84, v102
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_mul_f32_e32 v49, v86, v99
	v_add_f32_e32 v169, v169, v46
	v_fmac_f32_e32 v45, v48, v87
	v_mul_f32_e32 v41, v50, v41
	v_mul_f32_e32 v42, v51, v42
	v_mul_f32_e32 v46, v84, v101
	v_mul_f32_e32 v48, v84, v103
	v_fmac_f32_e32 v47, v49, v87
	v_mul_f32_e32 v49, v84, v96
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v46, v57
	v_fmac_f32_e32 v42, v48, v57
	v_add_f32_e32 v166, v166, v45
	v_mul_f32_e32 v45, v84, v97
	v_mul_f32_e32 v43, v49, v43
	v_add_f32_e32 v164, v164, v41
	v_add_f32_e32 v165, v165, v42
	v_mul_f32_e32 v41, v82, v100
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v42, v82, v102
	v_cvt_f32_i32_e32 v38, v38
	v_fmac_f32_e32 v43, v45, v57
	v_cvt_f32_i32_e32 v39, v39
	v_mul_f32_e32 v37, v41, v37
	v_mul_f32_e32 v41, v82, v101
	v_mul_f32_e32 v38, v42, v38
	v_mul_f32_e32 v42, v82, v96
	v_add_f32_e32 v162, v162, v43
	v_mul_f32_e32 v43, v82, v103
	v_fmac_f32_e32 v37, v41, v59
	v_mul_f32_e32 v41, v82, v98
	v_cvt_f32_i32_e32 v40, v40
	v_mul_f32_e32 v39, v42, v39
	v_mul_f32_e32 v42, v82, v97
	v_fmac_f32_e32 v38, v43, v59
	v_add_f32_e32 v160, v160, v37
	v_mul_f32_e32 v37, v41, v40
	v_mul_f32_e32 v40, v82, v99
	v_fmac_f32_e32 v39, v42, v59
	v_mul_f32_e32 v41, v72, v100
	v_mul_f32_e32 v42, v72, v102
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v161, v161, v38
	v_fmac_f32_e32 v37, v40, v59
	v_add_f32_e32 v158, v158, v39
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v38, v72, v101
	v_mul_f32_e32 v39, v72, v103
	v_mul_f32_e32 v40, v72, v96
	v_cvt_f32_i32_e32 v35, v35
	v_add_f32_e32 v159, v159, v37
	v_fmac_f32_e32 v33, v38, v56
	v_fmac_f32_e32 v34, v39, v56
	v_mul_f32_e32 v37, v72, v97
	v_mul_f32_e32 v35, v40, v35
	s_wait_dscnt 0x7
	v_mul_f32_e32 v39, v86, v94
	s_wait_dscnt 0x6
	v_mul_f32_e32 v40, v86, v92
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_add_f32_e32 v156, v156, v33
	v_fmac_f32_e32 v35, v37, v56
	v_mul_f32_e32 v33, v86, v95
	v_mul_f32_e32 v29, v39, v29
	v_mul_f32_e32 v30, v40, v30
	v_mul_f32_e32 v37, v86, v93
	v_add_f32_e32 v157, v157, v34
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v29, v33, v87
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v86, v90
	v_fmac_f32_e32 v30, v37, v87
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v86, v88
	v_cvt_f32_i32_e32 v32, v32
	v_add_f32_e32 v152, v152, v29
	v_mul_f32_e32 v29, v33, v31
	v_add_f32_e32 v153, v153, v30
	v_mul_f32_e32 v30, v86, v91
	v_mul_f32_e32 v31, v34, v32
	v_mul_f32_e32 v32, v84, v94
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v33, v86, v89
	v_fmac_f32_e32 v29, v30, v87
	v_mul_f32_e32 v30, v84, v92
	v_cvt_f32_i32_e32 v26, v26
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v32, v84, v95
	v_fmac_f32_e32 v31, v33, v87
	v_add_f32_e32 v150, v150, v29
	v_mul_f32_e32 v26, v30, v26
	v_mul_f32_e32 v29, v84, v93
	v_fmac_f32_e32 v25, v32, v57
	v_mul_f32_e32 v30, v84, v90
	v_mul_f32_e32 v32, v84, v88
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_add_f32_e32 v151, v151, v31
	v_fmac_f32_e32 v26, v29, v57
	v_add_f32_e32 v148, v148, v25
	v_mul_f32_e32 v25, v30, v27
	v_mul_f32_e32 v27, v32, v28
	v_mul_f32_e32 v28, v84, v91
	v_mul_f32_e32 v30, v82, v94
	v_mul_f32_e32 v31, v82, v92
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v29, v84, v89
	v_add_f32_e32 v149, v149, v26
	v_fmac_f32_e32 v25, v28, v57
	v_mul_f32_e32 v21, v30, v21
	v_mul_f32_e32 v22, v31, v22
	v_mul_f32_e32 v26, v82, v95
	v_mul_f32_e32 v28, v82, v93
	v_fmac_f32_e32 v27, v29, v57
	v_mul_f32_e32 v29, v82, v90
	v_cvt_f32_i32_e32 v23, v23
	v_fmac_f32_e32 v21, v26, v59
	v_fmac_f32_e32 v22, v28, v59
	v_add_f32_e32 v146, v146, v25
	v_mul_f32_e32 v25, v82, v91
	v_mul_f32_e32 v23, v29, v23
	v_add_f32_e32 v144, v144, v21
	v_add_f32_e32 v145, v145, v22
	v_mul_f32_e32 v21, v72, v94
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v72, v92
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v23, v25, v59
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v72, v95
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v72, v90
	v_add_f32_e32 v142, v142, v23
	v_mul_f32_e32 v23, v72, v93
	v_fmac_f32_e32 v17, v21, v56
	v_mul_f32_e32 v21, v72, v88
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v72, v91
	v_fmac_f32_e32 v18, v23, v56
	v_add_f32_e32 v140, v140, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v72, v89
	v_fmac_f32_e32 v19, v22, v56
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v86, v80
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v86, v78
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v141, v141, v18
	v_fmac_f32_e32 v17, v20, v56
	v_add_f32_e32 v138, v138, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v86, v81
	v_mul_f32_e32 v19, v86, v79
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v86, v74
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v139, v139, v17
	v_fmac_f32_e32 v13, v18, v87
	v_fmac_f32_e32 v14, v19, v87
	v_mul_f32_e32 v17, v86, v75
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v84, v80
	v_mul_f32_e32 v20, v84, v78
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v136, v136, v13
	v_fmac_f32_e32 v15, v17, v87
	v_mul_f32_e32 v13, v84, v81
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v84, v79
	v_add_f32_e32 v137, v137, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v57
	v_mul_f32_e32 v13, v84, v74
	v_fmac_f32_e32 v10, v17, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v84, v76
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v132, v132, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v133, v133, v10
	v_mul_f32_e32 v10, v84, v75
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v82, v80
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v57
	v_mul_f32_e32 v10, v82, v78
	v_mul_f32_e32 v13, v84, v77
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v82, v81
	v_add_f32_e32 v130, v130, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v82, v79
	v_mul_f32_e32 v10, v82, v74
	v_fmac_f32_e32 v5, v12, v59
	v_mul_f32_e32 v12, v82, v76
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v57
	v_fmac_f32_e32 v6, v9, v59
	v_add_f32_e32 v128, v128, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v82, v75
	v_mul_f32_e32 v9, v82, v77
	v_mul_f32_e32 v105, v116, v84
	v_mul_f32_e32 v50, v84, v98
	v_cvt_f32_i32_e32 v44, v44
	v_mul_f32_e32 v41, v72, v98
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v30, v82, v88
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v21, v86, v76
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v131, v131, v11
	v_mul_f32_e32 v10, v72, v80
	v_mul_f32_e32 v11, v72, v78
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v59
	v_fmac_f32_e32 v7, v9, v59
	v_mul_f32_e32 v8, v72, v74
	v_mul_f32_e32 v9, v72, v76
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v64, v105, v107
	v_mul_f32_e32 v85, v117, v84
	v_mul_f32_e32 v44, v50, v44
	v_mul_f32_e32 v46, v84, v99
	v_mul_f32_e32 v36, v41, v36
	v_mul_f32_e32 v38, v72, v99
	v_mul_f32_e32 v24, v30, v24
	v_mul_f32_e32 v26, v82, v89
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v86, v77
	v_add_f32_e32 v129, v129, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v72, v81
	v_mul_f32_e32 v10, v72, v79
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v72, v75
	v_mul_f32_e32 v9, v72, v77
	v_fmac_f32_e32 v64, v85, v57
	v_fmac_f32_e32 v44, v46, v57
	v_fmac_f32_e32 v36, v38, v56
	v_fmac_f32_e32 v24, v26, v59
	v_fmac_f32_e32 v16, v18, v87
	v_fmac_f32_e32 v1, v6, v56
	v_fmac_f32_e32 v2, v10, v56
	v_fmac_f32_e32 v3, v8, v56
	v_fmac_f32_e32 v4, v9, v56
	v_add_f32_e32 v180, v180, v64
	v_add_f32_e32 v175, v175, v55
	v_add_f32_e32 v167, v167, v47
	v_add_f32_e32 v163, v163, v44
	v_add_f32_e32 v154, v154, v35
	v_add_f32_e32 v155, v155, v36
	v_add_f32_e32 v147, v147, v27
	v_add_f32_e32 v143, v143, v24
	v_add_f32_e32 v134, v134, v15
	v_add_f32_e32 v135, v135, v16
	v_add_f32_e32 v127, v127, v5
	v_add_f32_e32 v126, v126, v7
	v_add_f32_e32 v123, v123, v1
	v_add_f32_e32 v125, v125, v2
	v_add_f32_e32 v67, v67, v3
	v_add_f32_e32 v124, v124, v4
	s_xor_b64 s[38:39], s[36:37], -1
	s_mov_b32 s43, 1
	s_mov_b64 s[36:37], 0
	s_and_b64 vcc, exec, s[38:39]
	s_mov_b64 s[38:39], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
.LBB1_16:                               ;   Parent Loop BB1_14 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s26, s43, s29
	s_lshl_b32 s44, s43, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[46:47], s[26:27], s[10:11]
	s_mov_b32 s45, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[46:47], s[46:47], 0x48
	s_add_nc_u64 s[44:45], s[34:35], s[44:45]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[46:47], s[14:15], s[46:47]
	v_add_nc_u32_e32 v78, s40, v191
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[48:49], s46, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s47, 0, s[48:49]
	v_add_co_u32 v3, s[48:49], s44, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s45, 0, s[48:49]
	v_add_co_u32 v5, s[48:49], s46, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s47, 0, s[48:49]
	v_add_co_u32 v7, s[46:47], s44, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s45, 0, s[46:47]
	global_load_b64 v[110:111], v[1:2], off offset:40
	global_load_b64 v[108:109], v[3:4], off offset:40
	global_load_b64 v[106:107], v[5:6], off offset:40
	global_load_b64 v[104:105], v[7:8], off offset:40
	v_add_nc_u32_e32 v76, s41, v190
	ds_load_2addr_stride64_b32 v[1:2], v78 offset1:2
	ds_load_2addr_stride64_b32 v[3:4], v76 offset1:2
	ds_load_2addr_stride64_b32 v[72:73], v76 offset0:4 offset1:6
	ds_load_2addr_stride64_b32 v[74:75], v78 offset0:4 offset1:6
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v1, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v1, v4, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v1, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v1, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v2, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v2, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v2, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v2, v73, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v74, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v74, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v74, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v74, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v75, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v75, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v75, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v75, v73, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v78 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[74:75], v76 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[76:77], v76 offset0:5 offset1:7
	ds_load_2addr_stride64_b32 v[78:79], v78 offset0:5 offset1:7
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v74, 0x2000, v183
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v110, s[0:1]
	v_add_nc_u32_e32 v73, 0x4000, v183
	v_cndmask_b32_e64 v75, 0, v111, s[0:1]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v74, v108, v109 offset1:16
	ds_store_2addr_b32 v73, v72, v75 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v106, s[2:3]
	v_cndmask_b32_e64 v73, 0, v107, s[2:3]
	v_add_nc_u32_e32 v74, 0x4000, v184
	v_add_nc_u32_e32 v75, 0x2000, v184
	ds_store_2addr_b32 v74, v72, v73 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v75, v104, v105 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[40:41], s[38:39], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[40:41]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_18
; %bb.17:                               ;   in Loop: Header=BB1_16 Depth=2
	s_add_co_i32 s26, s26, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[44:45], s[26:27], s[10:11]
	s_add_co_i32 s26, s43, s28
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[44:45], s[44:45], 0x48
	s_mul_u64 s[46:47], s[26:27], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[44:45], s[14:15], s[44:45]
	s_xor_b32 s43, s43, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v189, s[44:45]
	v_add_co_u32 v72, s[48:49], s44, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s45, 0, s[48:49]
	s_add_nc_u64 s[46:47], s[12:13], s[46:47]
	s_lshl_b32 s48, s43, 6
	s_mov_b32 s49, s27
	v_add_co_u32 v76, vcc, v76, v196
	s_mulk_i32 s26, 0x88
	v_add_co_u32 v74, s[50:51], s44, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[46:47], s[46:47], s[48:49]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc
	v_add_co_u32 v82, vcc, v65, s26
	v_add_co_ci_u32_e64 v75, null, s45, 0, s[50:51]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s[44:45], s46, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s47, 0, s[44:45]
	v_add_co_u32 v80, s[44:45], s46, v71
	s_lshl_b32 s26, s43, 2
	v_add_co_ci_u32_e64 v81, null, s47, 0, s[44:45]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc, v82, s26
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc
	s_clause 0x1
	global_load_b64 v[110:111], v[72:73], off offset:8
	global_load_b64 v[106:107], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[108:109], v[78:79], off offset:8
	global_load_b64 v[104:105], v[80:81], off offset:8
	global_load_b32 v197, v[76:77], off
	global_load_b32 v198, v[82:83], off
.LBB1_18:                               ; %.preheader532.i.i
                                        ;   in Loop: Header=BB1_16 Depth=2
	v_add_nc_u32_e32 v80, 0, v191
	v_add_nc_u32_e32 v81, 0, v190
	s_xor_b64 s[40:41], s[40:41], -1
	s_and_b64 s[44:45], s[36:37], exec
	s_cselect_b32 s26, s21, 0x3400
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:68 offset1:70
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:36 offset1:38
	s_cselect_b32 s43, s33, 0x3c00
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:69 offset1:71
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:37 offset1:39
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s43, v194
	v_add_nc_u32_e32 v72, s26, v195
	s_and_not1_b64 vcc, exec, s[40:41]
	s_movk_i32 s40, 0x2000
	s_movk_i32 s41, 0x4000
	ds_load_2addr_b32 v[116:117], v76 offset1:1
	ds_load_2addr_b32 v[118:119], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[112:113], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[86:87], v72 offset1:1
	ds_load_2addr_b32 v[84:85], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[82:83], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[100:101], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[102:103], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[96:97], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[98:99], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[94:95], v76 offset0:64 offset1:65
	ds_load_2addr_b32 v[92:93], v76 offset0:66 offset1:67
	ds_load_2addr_b32 v[90:91], v76 offset0:68 offset1:69
	ds_load_2addr_b32 v[88:89], v76 offset0:70 offset1:71
	ds_load_2addr_b32 v[80:81], v76 offset0:96 offset1:97
	ds_load_2addr_b32 v[78:79], v76 offset0:98 offset1:99
	ds_load_2addr_b32 v[74:75], v76 offset0:100 offset1:101
	ds_load_2addr_b32 v[76:77], v76 offset0:102 offset1:103
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_15
; %bb.19:                               ; %.preheader533.i.i
                                        ;   in Loop: Header=BB1_16 Depth=2
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v110, 0, v110, s[0:1]
	v_cndmask_b32_e64 v111, 0, v111, s[0:1]
	v_add_nc_u32_e32 v199, 0x1000, v183
	s_and_b64 s[38:39], s[38:39], exec
	s_cselect_b32 s26, s21, 0x3400
	ds_store_2addr_b32 v183, v110, v111 offset1:16
	s_wait_loadcnt 0x3
	ds_store_2addr_b32 v199, v108, v109 offset1:16
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v108, v193, v198
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v111, s26, v192
	s_cselect_b32 s26, s33, 0x3c00
	v_cndmask_b32_e64 v106, 0, v106, s[2:3]
	v_cndmask_b32_e64 v107, 0, v107, s[2:3]
	v_cvt_f32_f16_e32 v108, v108.l
	v_cndmask_b32_e64 v110, 0, v197, s[4:5]
	v_add_nc_u32_e32 v109, 0x1000, v184
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v199, s26, v192
	s_mov_b32 s41, 0
	v_cndmask_b32_e64 v108, 0, v108, s[6:7]
	s_movk_i32 s40, 0x1000
	ds_store_2addr_b32 v184, v106, v107 offset1:16
	ds_store_2addr_b32 v109, v104, v105 offset1:16
	ds_store_b32 v111, v110
	ds_store_b32 v199, v108
	s_branch .LBB1_15
.LBB1_20:                               ; %Flow2273
	v_mov_b32_e32 v6, v182
.LBB1_21:                               ; %._crit_edge604.i.i
	v_and_b32_e32 v4, 15, v0
	v_mul_u32_u24_e32 v1, 0x500, v120
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s27, s8, 31
	s_mov_b32 s26, s8
	v_lshlrev_b32_e32 v2, 2, v4
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[26:27], s[18:19]
	s_ashr_i32 s21, s20, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[20:21], 2
	v_add3_u32 v9, 0, v1, v2
	v_lshrrev_b32_e32 v2, 4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	s_mov_b32 s7, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mad_i32_i24 v1, 0x50, v6, v9
	v_lshlrev_b32_e32 v3, 2, v2
	s_add_co_i32 s0, s20, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s0, s8
	ds_store_2addr_b32 v1, v187, v188 offset1:20
	ds_store_2addr_b32 v1, v185, v186 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v4
	s_cselect_b64 s[0:1], -1, 0
	s_add_co_i32 s2, s18, 0x80
	v_or_b32_e32 v7, s18, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s2, s10
	v_add3_u32 v3, 0, v1, v3
	v_or_b32_e32 v1, s20, v4
	v_mul_lo_u32 v8, s8, v2
	s_cselect_b64 s[2:3], -1, 0
	s_mov_b32 s6, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[28:29], s[0:1], s[2:3]
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	v_cmp_gt_i32_e64 s[2:3], s10, v7
	s_and_b64 vcc, exec, s[28:29]
	s_mov_b64 s[30:31], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v5, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_25
; %bb.22:
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_24
; %bb.23:
	v_mad_co_i64_i32 v[10:11], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v2, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	global_load_b32 v2, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v5, v2
	global_store_b32 v[10:11], v2, off
.LBB1_24:                               ; %Flow2269
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[30:31], 0
.LBB1_25:                               ; %Flow2270
	v_add_lshl_u32 v4, v8, v4, 2
	s_and_not1_b64 vcc, exec, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_27
; %bb.26:
	buffer_load_b32 v2, v4, s[4:7], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v5, v2
	buffer_store_b32 v2, v4, s[4:7], null offen
.LBB1_27:
	ds_load_b32 v10, v3 offset:1280
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v5, 0, 1, s[28:29]
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	v_or_b32_e32 v8, 64, v7
	s_and_not1_b64 vcc, exec, s[28:29]
	s_mov_b64 s[2:3], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_31
; %bb.28:
	v_cmp_gt_i32_e32 vcc, s10, v8
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_30
; %bb.29:
	v_mad_co_i64_i32 v[11:12], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc
	global_load_b32 v2, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[11:12], v2, off
.LBB1_30:                               ; %Flow2267
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_31:                               ; %Flow2268
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_33
; %bb.32:
	s_lshl_b32 s0, s8, 8
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_33:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	v_or_b32_e32 v15, 64, v1
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_37
; %bb.34:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_36
; %bb.35:
	v_mad_co_i64_i32 v[11:12], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc
	global_load_b32 v2, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[11:12], v2, off offset:256
.LBB1_36:                               ; %Flow2265
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_37:                               ; %Flow2266
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_39
; %bb.38:
	s_movk_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_39:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_43
; %bb.40:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_42
; %bb.41:
	v_mad_co_i64_i32 v[11:12], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc
	global_load_b32 v2, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[11:12], v2, off offset:256
.LBB1_42:                               ; %Flow2263
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_43:                               ; %Flow2264
	v_mul_i32_i24_e32 v2, 0x50, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_45
; %bb.44:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v6, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v10, v6
	buffer_store_b32 v6, v4, s[4:7], s0 offen
.LBB1_45:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v6, v9, v2
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	v_or_b32_e32 v9, 16, v7
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v180, v181 offset1:20
	ds_store_2addr_b32 v6, v178, v179 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v10, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_49
; %bb.46:
	v_cmp_gt_i32_e32 vcc, s10, v9
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_48
; %bb.47:
	v_mad_co_i64_i32 v[11:12], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc
	global_load_b32 v2, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[11:12], v2, off
.LBB1_48:                               ; %Flow2261
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_49:                               ; %Flow2262
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_51
; %bb.50:
	s_lshl_b32 s0, s8, 6
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_51:
	ds_load_b32 v11, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	s_wait_dscnt 0x1
	v_or_b32_e32 v10, 0x50, v7
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_55
; %bb.52:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v10
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_54
; %bb.53:
	v_mad_co_i64_i32 v[12:13], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc, v2, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v17, vcc
	global_load_b32 v2, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	global_store_b32 v[12:13], v2, off
.LBB1_54:                               ; %Flow2259
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_55:                               ; %Flow2260
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_57
; %bb.56:
	s_mul_i32 s0, s8, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_57:
	s_wait_dscnt 0x0
	ds_load_b32 v11, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_61
; %bb.58:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_60
; %bb.59:
	v_mad_co_i64_i32 v[12:13], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc, v2, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v17, vcc
	global_load_b32 v2, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	global_store_b32 v[12:13], v2, off offset:256
.LBB1_60:                               ; %Flow2257
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_61:                               ; %Flow2258
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_63
; %bb.62:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_63:
	s_wait_dscnt 0x0
	ds_load_b32 v11, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_67
; %bb.64:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_co_i64_i32 v[12:13], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc, v2, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v17, vcc
	global_load_b32 v2, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	global_store_b32 v[12:13], v2, off offset:256
.LBB1_66:                               ; %Flow2255
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_67:                               ; %Flow2256
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_69
; %bb.68:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_69:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	v_or_b32_e32 v11, 32, v7
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v176, v177 offset1:20
	ds_store_2addr_b32 v6, v174, v175 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v12, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_73
; %bb.70:
	v_cmp_gt_i32_e32 vcc, s10, v11
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_72
; %bb.71:
	v_mad_co_i64_i32 v[13:14], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, vcc, v2, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v14, v17, vcc
	global_load_b32 v2, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[13:14], v2, off
.LBB1_72:                               ; %Flow2253
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_73:                               ; %Flow2254
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_75
; %bb.74:
	s_lshl_b32 s0, s8, 7
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_75:
	ds_load_b32 v13, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	s_wait_dscnt 0x1
	v_or_b32_e32 v12, 0x60, v7
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_79
; %bb.76:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v12
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_78
; %bb.77:
	v_mad_co_i64_i32 v[16:17], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v14, v19, vcc
	global_load_b32 v2, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[16:17], v2, off
.LBB1_78:                               ; %Flow2251
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_79:                               ; %Flow2252
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_81
; %bb.80:
	s_mul_i32 s0, s8, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_81:
	s_wait_dscnt 0x0
	ds_load_b32 v13, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_85
; %bb.82:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_84
; %bb.83:
	v_mad_co_i64_i32 v[16:17], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v14, v19, vcc
	global_load_b32 v2, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[16:17], v2, off offset:256
.LBB1_84:                               ; %Flow2249
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_85:                               ; %Flow2250
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_87
; %bb.86:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_87:
	s_wait_dscnt 0x0
	ds_load_b32 v13, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_91
; %bb.88:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_90
; %bb.89:
	v_mad_co_i64_i32 v[16:17], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v14, v19, vcc
	global_load_b32 v2, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[16:17], v2, off offset:256
.LBB1_90:                               ; %Flow2247
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_91:                               ; %Flow2248
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_93
; %bb.92:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_93:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s8, v1
	v_or_b32_e32 v14, 48, v7
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v172, v173 offset1:20
	ds_store_2addr_b32 v6, v170, v171 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_97
; %bb.94:
	v_cmp_gt_i32_e32 vcc, s10, v14
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_96
; %bb.95:
	v_mad_co_i64_i32 v[16:17], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v2, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[16:17], v2, off
.LBB1_96:                               ; %Flow2245
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_97:                               ; %Flow2246
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_99
; %bb.98:
	s_mul_i32 s0, s8, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_99:
	ds_load_b32 v16, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v13, 0x70, v7
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_103
; %bb.100:
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_102
; %bb.101:
	v_mad_co_i64_i32 v[17:18], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off
.LBB1_102:                              ; %Flow2243
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_103:                              ; %Flow2244
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_105
; %bb.104:
	s_mul_i32 s0, s8, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_105:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_109
; %bb.106:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_108
; %bb.107:
	v_mad_co_i64_i32 v[17:18], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:256
.LBB1_108:                              ; %Flow2241
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_109:                              ; %Flow2242
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_111
; %bb.110:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_111:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_115
; %bb.112:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_114
; %bb.113:
	v_mad_co_i64_i32 v[17:18], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v15, v20, vcc
	global_load_b32 v2, v[17:18], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:256
.LBB1_114:                              ; %Flow2239
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_115:                              ; %Flow2240
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_117
; %bb.116:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_117:                              ; %.preheader529.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	v_or_b32_e32 v15, 16, v1
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v168, v169 offset1:20
	ds_store_2addr_b32 v6, v166, v167 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_121
; %bb.118:
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_120
; %bb.119:
	v_mad_co_i64_i32 v[17:18], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:64
.LBB1_120:                              ; %Flow2237
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_121:                              ; %Flow2238
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_123
; %bb.122:
	s_mov_b32 s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_123:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_127
; %bb.124:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_126
; %bb.125:
	v_mad_co_i64_i32 v[17:18], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:64
.LBB1_126:                              ; %Flow2235
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_127:                              ; %Flow2236
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_129
; %bb.128:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_129:
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_dscnt 0x1
	v_or_b32_e32 v16, 0x50, v1
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_133
; %bb.130:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v16
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_132
; %bb.131:
	v_mad_co_i64_i32 v[18:19], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_132:                              ; %Flow2233
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_133:                              ; %Flow2234
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_135
; %bb.134:
	s_movk_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_135:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_139
; %bb.136:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_138
; %bb.137:
	v_mad_co_i64_i32 v[18:19], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_138:                              ; %Flow2231
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_139:                              ; %Flow2232
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_141
; %bb.140:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_141:                              ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v164, v165 offset1:20
	ds_store_2addr_b32 v6, v162, v163 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_145
; %bb.142:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_144
; %bb.143:
	v_mad_co_i64_i32 v[18:19], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_144:                              ; %Flow2229
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_145:                              ; %Flow2230
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_147
; %bb.146:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_147:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_151
; %bb.148:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_150
; %bb.149:
	v_mad_co_i64_i32 v[18:19], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_150:                              ; %Flow2227
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_151:                              ; %Flow2228
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_153
; %bb.152:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_153:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_157
; %bb.154:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_156
; %bb.155:
	v_mad_co_i64_i32 v[18:19], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_156:                              ; %Flow2225
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_157:                              ; %Flow2226
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_159
; %bb.158:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_159:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_163
; %bb.160:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_162
; %bb.161:
	v_mad_co_i64_i32 v[18:19], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_162:                              ; %Flow2223
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_163:                              ; %Flow2224
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_165
; %bb.164:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_165:                              ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v160, v161 offset1:20
	ds_store_2addr_b32 v6, v158, v159 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_169
; %bb.166:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_168
; %bb.167:
	v_mad_co_i64_i32 v[18:19], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_168:                              ; %Flow2221
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_169:                              ; %Flow2222
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_171
; %bb.170:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_171:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_175
; %bb.172:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_174
; %bb.173:
	v_mad_co_i64_i32 v[18:19], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_174:                              ; %Flow2219
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_175:                              ; %Flow2220
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_177
; %bb.176:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_177:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_181
; %bb.178:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_180
; %bb.179:
	v_mad_co_i64_i32 v[18:19], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_180:                              ; %Flow2217
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_181:                              ; %Flow2218
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_183
; %bb.182:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_183:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_187
; %bb.184:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_186
; %bb.185:
	v_mad_co_i64_i32 v[18:19], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:320
.LBB1_186:                              ; %Flow2215
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_187:                              ; %Flow2216
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_189
; %bb.188:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_189:                              ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v156, v157 offset1:20
	ds_store_2addr_b32 v6, v154, v155 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_193
; %bb.190:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_192
; %bb.191:
	v_mad_co_i64_i32 v[18:19], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_192:                              ; %Flow2213
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_193:                              ; %Flow2214
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_195
; %bb.194:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_195:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_199
; %bb.196:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_198
; %bb.197:
	v_mad_co_i64_i32 v[18:19], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v15, v21, vcc
	global_load_b32 v2, v[18:19], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:64
.LBB1_198:                              ; %Flow2211
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_199:                              ; %Flow2212
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_201
; %bb.200:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_201:
	ds_load_b32 v15, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_205
; %bb.202:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_204
; %bb.203:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[17:18], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[17:18], v2, off offset:320
.LBB1_204:                              ; %Flow2209
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_205:                              ; %Flow2210
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_207
; %bb.206:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_207:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_211
; %bb.208:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_210
; %bb.209:
	v_mad_co_i64_i32 v[16:17], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v2, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[16:17], v2, off offset:320
.LBB1_210:                              ; %Flow2207
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_211:                              ; %Flow2208
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_213
; %bb.212:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_213:                              ; %.preheader529.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	v_or_b32_e32 v15, 32, v1
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v152, v153 offset1:20
	ds_store_2addr_b32 v6, v150, v151 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_217
; %bb.214:
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_216
; %bb.215:
	v_mad_co_i64_i32 v[17:18], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:128
.LBB1_216:                              ; %Flow2205
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_217:                              ; %Flow2206
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_219
; %bb.218:
	s_movk_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_219:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_223
; %bb.220:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_222
; %bb.221:
	v_mad_co_i64_i32 v[17:18], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	global_store_b32 v[17:18], v2, off offset:128
.LBB1_222:                              ; %Flow2203
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_223:                              ; %Flow2204
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_225
; %bb.224:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_bitset1_b32 s0, 7
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v16, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_225:
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_dscnt 0x1
	v_or_b32_e32 v16, 0x60, v1
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB1_229
; %bb.226:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v16
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_228
; %bb.227:
	v_mad_co_i64_i32 v[18:19], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_228:                              ; %Flow2201
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_229:                              ; %Flow2202
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_231
; %bb.230:
	s_movk_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_231:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_235
; %bb.232:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_234
; %bb.233:
	v_mad_co_i64_i32 v[18:19], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_234:                              ; %Flow2199
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_235:                              ; %Flow2200
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_237
; %bb.236:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_237:                              ; %.preheader.1.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v148, v149 offset1:20
	ds_store_2addr_b32 v6, v146, v147 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_241
; %bb.238:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_240
; %bb.239:
	v_mad_co_i64_i32 v[18:19], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_240:                              ; %Flow2197
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_241:                              ; %Flow2198
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_243
; %bb.242:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_243:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_247
; %bb.244:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_246
; %bb.245:
	v_mad_co_i64_i32 v[18:19], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_246:                              ; %Flow2195
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_247:                              ; %Flow2196
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_249
; %bb.248:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_249:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_253
; %bb.250:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_252
; %bb.251:
	v_mad_co_i64_i32 v[18:19], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_252:                              ; %Flow2193
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_253:                              ; %Flow2194
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_255
; %bb.254:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_255:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_259
; %bb.256:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_258
; %bb.257:
	v_mad_co_i64_i32 v[18:19], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_258:                              ; %Flow2191
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_259:                              ; %Flow2192
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_261
; %bb.260:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_261:                              ; %.preheader.2.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v144, v145 offset1:20
	ds_store_2addr_b32 v6, v142, v143 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_265
; %bb.262:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_264
; %bb.263:
	v_mad_co_i64_i32 v[18:19], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_264:                              ; %Flow2189
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_265:                              ; %Flow2190
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_267
; %bb.266:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_267:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_271
; %bb.268:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_270
; %bb.269:
	v_mad_co_i64_i32 v[18:19], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_270:                              ; %Flow2187
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_271:                              ; %Flow2188
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_273
; %bb.272:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_273:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_277
; %bb.274:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_276
; %bb.275:
	v_mad_co_i64_i32 v[18:19], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_276:                              ; %Flow2185
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_277:                              ; %Flow2186
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_279
; %bb.278:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_279:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_283
; %bb.280:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_282
; %bb.281:
	v_mad_co_i64_i32 v[18:19], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:384
.LBB1_282:                              ; %Flow2183
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_283:                              ; %Flow2184
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_285
; %bb.284:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_285:                              ; %.preheader.3.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v140, v141 offset1:20
	ds_store_2addr_b32 v6, v138, v139 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v17, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_289
; %bb.286:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_288
; %bb.287:
	v_mad_co_i64_i32 v[18:19], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v19, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_288:                              ; %Flow2181
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_289:                              ; %Flow2182
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_291
; %bb.290:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_291:
	s_wait_dscnt 0x0
	ds_load_b32 v17, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_295
; %bb.292:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_294
; %bb.293:
	v_mad_co_i64_i32 v[18:19], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v15, v21, vcc
	global_load_b32 v2, v[18:19], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:128
.LBB1_294:                              ; %Flow2179
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_295:                              ; %Flow2180
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_297
; %bb.296:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_297:
	ds_load_b32 v15, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_301
; %bb.298:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_300
; %bb.299:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[17:18], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[17:18], v2, off offset:384
.LBB1_300:                              ; %Flow2177
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_301:                              ; %Flow2178
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_303
; %bb.302:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_303:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_307
; %bb.304:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_306
; %bb.305:
	v_mad_co_i64_i32 v[16:17], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v2, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v2, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[16:17], v2, off offset:384
.LBB1_306:                              ; %Flow2175
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_307:                              ; %Flow2176
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_309
; %bb.308:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_309:                              ; %.preheader529.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	v_or_b32_e32 v16, 48, v1
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v136, v137 offset1:20
	ds_store_2addr_b32 v6, v134, v135 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_313
; %bb.310:
	v_cmp_gt_i32_e32 vcc, s8, v16
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_312
; %bb.311:
	v_mad_co_i64_i32 v[17:18], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[17:18], v2, off offset:192
.LBB1_312:                              ; %Flow2173
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB1_313:                              ; %Flow2174
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_315
; %bb.314:
	s_movk_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_315:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_319
; %bb.316:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_318
; %bb.317:
	v_mad_co_i64_i32 v[17:18], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v2, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[17:18], v2, off offset:192
.LBB1_318:                              ; %Flow2171
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_319:                              ; %Flow2172
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_321
; %bb.320:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_321:
	ds_load_b32 v17, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x70, v1
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_325
; %bb.322:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_324
; %bb.323:
	v_mad_co_i64_i32 v[18:19], null, s26, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[20:21], 2, v[1:2]
	v_lshlrev_b64_e32 v[18:19], 2, v[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v19, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v18, vcc, v2, v20
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v7, v21, vcc
	global_load_b32 v2, v[18:19], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	global_store_b32 v[18:19], v2, off offset:448
.LBB1_324:                              ; %Flow2169
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_325:                              ; %Flow2170
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_327
; %bb.326:
	s_movk_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v17, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_327:
	ds_load_b32 v7, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_331
; %bb.328:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_330
; %bb.329:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[17:18], null, s26, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v8, v20, vcc
	global_load_b32 v2, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[17:18], v2, off offset:448
.LBB1_330:                              ; %Flow2167
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_331:                              ; %Flow2168
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_333
; %bb.332:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_333:                              ; %.preheader.1.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v132, v133 offset1:20
	ds_store_2addr_b32 v6, v130, v131 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_337
; %bb.334:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_336
; %bb.335:
	v_mad_co_i64_i32 v[17:18], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v8, v20, vcc
	global_load_b32 v2, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[17:18], v2, off offset:192
.LBB1_336:                              ; %Flow2165
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_337:                              ; %Flow2166
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_339
; %bb.338:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_339:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_343
; %bb.340:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_342
; %bb.341:
	v_mad_co_i64_i32 v[17:18], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v2, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v8, v20, vcc
	global_load_b32 v2, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[17:18], v2, off offset:192
.LBB1_342:                              ; %Flow2163
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_343:                              ; %Flow2164
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_345
; %bb.344:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_345:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_349
; %bb.346:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_348
; %bb.347:
	v_mad_co_i64_i32 v[8:9], null, s26, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v18, vcc
	global_load_b32 v2, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:448
.LBB1_348:                              ; %Flow2161
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_349:                              ; %Flow2162
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_351
; %bb.350:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_351:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_355
; %bb.352:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_354
; %bb.353:
	v_mad_co_i64_i32 v[8:9], null, s26, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v18, vcc
	global_load_b32 v2, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:448
.LBB1_354:                              ; %Flow2159
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_355:                              ; %Flow2160
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_357
; %bb.356:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_357:                              ; %.preheader.2.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v128, v129 offset1:20
	ds_store_2addr_b32 v6, v127, v126 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_361
; %bb.358:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_360
; %bb.359:
	v_mad_co_i64_i32 v[8:9], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v18, vcc
	global_load_b32 v2, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:192
.LBB1_360:                              ; %Flow2157
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_361:                              ; %Flow2158
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_363
; %bb.362:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_363:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_367
; %bb.364:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_366
; %bb.365:
	v_mad_co_i64_i32 v[8:9], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v18, vcc
	global_load_b32 v2, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:192
.LBB1_366:                              ; %Flow2155
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_367:                              ; %Flow2156
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_369
; %bb.368:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_369:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_373
; %bb.370:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_372
; %bb.371:
	v_mad_co_i64_i32 v[8:9], null, s26, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	global_load_b32 v2, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:448
.LBB1_372:                              ; %Flow2153
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_373:                              ; %Flow2154
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_375
; %bb.374:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_375:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_379
; %bb.376:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_378
; %bb.377:
	v_mad_co_i64_i32 v[8:9], null, s26, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v2, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	global_load_b32 v2, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[8:9], v2, off offset:448
.LBB1_378:                              ; %Flow2151
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_379:                              ; %Flow2152
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_381
; %bb.380:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v7, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_381:                              ; %.preheader.3.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v123, v125 offset1:20
	ds_store_2addr_b32 v6, v67, v124 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_385
; %bb.382:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_384
; %bb.383:
	v_mad_co_i64_i32 v[7:8], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	global_load_b32 v2, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	global_store_b32 v[7:8], v2, off offset:192
.LBB1_384:                              ; %Flow2149
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_385:                              ; %Flow2150
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_387
; %bb.386:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_387:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_391
; %bb.388:
	v_cmp_gt_i32_e32 vcc, s8, v16
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_390
; %bb.389:
	v_mad_co_i64_i32 v[7:8], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	global_load_b32 v2, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	global_store_b32 v[7:8], v2, off offset:192
.LBB1_390:                              ; %Flow2147
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_391:                              ; %Flow2148
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_393
; %bb.392:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_393:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_397
; %bb.394:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_396
; %bb.395:
	v_mad_co_i64_i32 v[7:8], null, s26, v14, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	global_load_b32 v2, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	global_store_b32 v[7:8], v2, off offset:448
.LBB1_396:                              ; %Flow2145
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_397:                              ; %Flow2146
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_399
; %bb.398:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v2, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v6, v2
	buffer_store_b32 v2, v4, s[4:7], s0 offen
.LBB1_399:
	ds_load_b32 v3, v3 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_403
; %bb.400:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_402
; %bb.401:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[5:6], null, s26, v13, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc, s16, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v6, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, v5, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v6, v2, vcc
	global_load_b32 v5, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v3, v5
	global_store_b32 v[1:2], v5, off offset:448
.LBB1_402:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_403:                              ; %Flow2144
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_405
; %bb.404:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v1, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v3, v1
	buffer_store_b32 v1, v4, s[4:7], s0 offen
.LBB1_405:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b64 s[0:1], -1
	s_barrier_wait -1
	s_and_b64 vcc, exec, s[24:25]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_808
.LBB1_406:
	s_and_b64 vcc, exec, s[22:23]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_808
; %bb.407:                              ; %.preheader533.i.i17
	s_ashr_i32 s0, s9, 31
	s_add_co_i32 s7, s8, -1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s0, s0, 24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s19, s0, 8
	s_mov_b64 s[0:1], exec
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s6, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_411
; %bb.408:                              ; %.lr.ph.i.i136
	v_and_b32_e32 v1, 1, v0
	v_or_b32_e32 v4, s18, v122
	v_mov_b32_e32 v3, 0
	s_mov_b64 s[2:3], exec
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 2, v1
	v_cmpx_gt_i32_e64 s10, v4
	s_cbranch_execz .LBB1_410
; %bb.409:
	v_mad_co_i64_i32 v[3:4], null, 0x48, v4, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc, v3, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc
	global_load_b32 v3, v[3:4], off
.LBB1_410:                              ; %.preheader530.loopexit.i.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v6, s20, v122
	v_lshlrev_b32_e32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v4, s7, v6
	v_cmp_gt_i32_e32 vcc, s8, v6
	v_mad_co_i64_i32 v[4:5], null, s6, v4, s[12:13]
	global_load_b32 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v1, v4
	v_lshlrev_b32_e32 v4, 3, v122
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_add3_u32 v2, 0, v4, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v1, 0, v1, vcc
	ds_store_2addr_stride64_b32 v2, v3, v1 offset0:48 offset1:56
.LBB1_411:                              ; %Flow2413
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v1, 2, v0
	s_add_co_i32 s21, s10, -1
	v_lshrrev_b32_e32 v11, 5, v0
	v_lshlrev_b32_e32 v12, 7, v0
	v_and_b32_e32 v13, 60, v0
	v_add_nc_u32_e32 v9, s18, v1
	v_add_nc_u32_e32 v2, s20, v1
	v_and_b32_e32 v1, 3, v0
	v_bfe_u32 v14, v0, 1, 1
	v_or_b32_e32 v15, 8, v11
	v_add_nc_u32_e32 v10, 64, v9
	v_add_nc_u32_e32 v3, 64, v2
	v_lshlrev_b32_e32 v1, 3, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s21, v9
	v_min_i32_e32 v2, s7, v2
	v_min_i32_e32 v5, s21, v10
	v_min_i32_e32 v3, s7, v3
	v_and_or_b32 v12, 0x80, v12, v13
	v_and_or_b32 v11, v11, 6, v14
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s6, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s6, v3, v[1:2]
	v_and_or_b32 v13, v15, 14, v14
	v_lshl_or_b32 v11, v11, 8, v12
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	global_load_b64 v[1:2], v68, s[14:15] offset:8
	global_load_b64 v[3:4], v69, s[12:13] offset:8
	global_load_b64 v[5:6], v70, s[14:15] offset:8
	global_load_b64 v[7:8], v71, s[12:13] offset:8
	v_lshl_or_b32 v12, v13, 8, v12
	v_add_nc_u32_e32 v183, 0, v11
	v_cmp_gt_i32_e64 s[2:3], s10, v10
	s_cmp_gt_i32 s9, 0xff
	v_add_nc_u32_e32 v184, 0, v12
	v_add_nc_u32_e32 v9, 0x1000, v183
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v10, 0x1000, v184
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v1, 0, v1, s[0:1]
	v_cndmask_b32_e64 v2, 0, v2, s[0:1]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v5, 0, v5, s[2:3]
	v_cndmask_b32_e64 v6, 0, v6, s[2:3]
	ds_store_2addr_b32 v9, v3, v4 offset1:16
	ds_store_2addr_b32 v183, v1, v2 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v10, v7, v8 offset1:16
	ds_store_2addr_b32 v184, v5, v6 offset1:16
	v_bfe_u32 v1, v0, 4, 2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_413
; %bb.412:                              ; %._crit_edge.._crit_edge599_crit_edge.i.i
	v_lshlrev_b32_e32 v178, 2, v1
	s_mov_b64 s[4:5], 0
	s_branch .LBB1_414
.LBB1_413:
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr178
.LBB1_414:                              ; %Flow2411
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v123, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_423
; %bb.415:                              ; %.preheader529.lr.ph.i.i
	v_add_nc_u32_e32 v2, s20, v122
	v_and_b32_e32 v3, 63, v0
	v_and_b32_e32 v7, 1, v0
	v_lshlrev_b32_e32 v4, 5, v0
	v_lshlrev_b32_e32 v178, 2, v1
	v_min_i32_e32 v6, s7, v2
	v_lshlrev_b32_e32 v1, 3, v122
	v_lshlrev_b32_e32 v3, 2, v3
	v_lshlrev_b32_e32 v8, 3, v0
	v_lshlrev_b32_e32 v9, 2, v7
	v_mad_co_u64_u32 v[65:66], null, s6, v6, s[12:13]
	v_ashrrev_i32_e32 v6, 31, v6
	v_add_nc_u32_e32 v5, s18, v122
	v_and_or_b32 v189, 0x800, v4, v3
	v_lshl_or_b32 v190, v121, 5, v3
	v_or_b32_e32 v3, v178, v121
	v_add3_u32 v121, 0, v1, v9
	v_and_b32_e32 v1, 0x278, v8
	v_mad_co_u64_u32 v[66:67], null, s6, v6, v[66:67]
	v_mov_b32_e32 v123, 0
	v_min_i32_e32 v122, s21, v5
	v_cmp_gt_i32_e64 s[4:5], s10, v5
	v_cmp_gt_i32_e64 s[6:7], s8, v2
	v_lshlrev_b32_e32 v191, 4, v7
	v_lshl_add_u32 v192, v3, 3, 0
	v_add_nc_u32_e32 v193, 0, v1
	v_lshlrev_b32_e32 v194, 2, v7
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	s_mov_b32 s23, 0
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s36, 0x1000
	s_movk_i32 s9, 0x3000
	s_movk_i32 s21, 0x3800
	s_mov_b32 s37, 0
	s_mov_b32 s24, s23
	s_branch .LBB1_417
.LBB1_416:                              ;   in Loop: Header=BB1_417 Depth=1
	s_and_b64 vcc, exec, s[26:27]
	s_mov_b32 s24, s33
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_423
.LBB1_417:                              ; %.preheader529.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_419 Depth 2
	s_mov_b32 s25, s23
	s_add_co_i32 s33, s24, 1
	s_mul_u64 s[28:29], s[24:25], 0x88
	s_lshl_b32 s25, s24, 1
	s_cmp_eq_u32 s33, s19
	s_mov_b64 s[34:35], 0
	s_cselect_b64 s[26:27], -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[28:29], s[12:13], s[28:29]
	s_mov_b64 s[30:31], -1
	s_mov_b32 s38, s23
	s_branch .LBB1_419
.LBB1_418:                              ;   in Loop: Header=BB1_419 Depth=2
	s_wait_loadcnt_dscnt 0x20f
	v_mul_f32_e32 v104, v116, v86
	v_cvt_f32_i32_e32 v61, v61
	v_cvt_f32_i32_e32 v87, v87
	v_mul_f32_e32 v105, v118, v86
	v_cvt_f32_i32_e32 v62, v62
	v_mul_f32_e32 v106, v117, v86
	v_mul_f32_e32 v61, v104, v61
	v_mul_f32_e32 v104, v119, v86
	v_mul_f32_e32 v107, v114, v86
	v_mul_f32_e32 v62, v105, v62
	v_mul_f32_e32 v105, v112, v86
	v_fmac_f32_e32 v61, v106, v87
	v_cvt_f32_i32_e32 v63, v63
	v_cvt_f32_i32_e32 v64, v64
	v_fmac_f32_e32 v62, v104, v87
	v_mul_f32_e32 v104, v113, v86
	v_add_f32_e32 v187, v187, v61
	v_mul_f32_e32 v61, v107, v63
	v_mul_f32_e32 v63, v105, v64
	v_mul_f32_e32 v64, v115, v86
	s_wait_dscnt 0xe
	v_mul_f32_e32 v106, v118, v84
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v107, v57
	v_fmac_f32_e32 v63, v104, v87
	v_fmac_f32_e32 v61, v64, v87
	v_cvt_f32_i32_e32 v57, v85
	v_mul_f32_e32 v58, v106, v58
	v_mul_f32_e32 v104, v119, v84
	v_add_f32_e32 v188, v188, v62
	v_add_f32_e32 v185, v185, v61
	v_mul_f32_e32 v61, v114, v84
	v_cvt_f32_i32_e32 v59, v59
	v_fmac_f32_e32 v58, v104, v57
	v_mul_f32_e32 v62, v112, v84
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v53, v53
	v_add_f32_e32 v186, v186, v63
	v_add_f32_e32 v182, v182, v58
	v_mul_f32_e32 v58, v61, v59
	v_mul_f32_e32 v59, v115, v84
	v_mul_f32_e32 v60, v62, v60
	s_wait_dscnt 0xd
	v_mul_f32_e32 v62, v116, v82
	v_mul_f32_e32 v61, v113, v84
	v_mul_f32_e32 v63, v118, v82
	v_fmac_f32_e32 v58, v59, v57
	v_cvt_f32_i32_e32 v59, v83
	v_cvt_f32_i32_e32 v54, v54
	v_mul_f32_e32 v53, v62, v53
	v_mul_f32_e32 v62, v117, v82
	v_fmac_f32_e32 v60, v61, v57
	v_add_f32_e32 v179, v179, v58
	v_mul_f32_e32 v54, v63, v54
	v_mul_f32_e32 v58, v119, v82
	v_fmac_f32_e32 v53, v62, v59
	v_mul_f32_e32 v61, v114, v82
	v_mul_f32_e32 v62, v112, v82
	v_cvt_f32_i32_e32 v55, v55
	v_cvt_f32_i32_e32 v56, v56
	v_add_f32_e32 v180, v180, v60
	v_fmac_f32_e32 v54, v58, v59
	v_add_f32_e32 v176, v176, v53
	v_mul_f32_e32 v53, v61, v55
	v_mul_f32_e32 v55, v62, v56
	v_mul_f32_e32 v56, v115, v82
	v_mul_f32_e32 v58, v113, v82
	s_wait_dscnt 0xc
	v_mul_f32_e32 v60, v116, v72
	v_mul_f32_e32 v61, v118, v72
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_fmac_f32_e32 v53, v56, v59
	v_fmac_f32_e32 v55, v58, v59
	v_cvt_f32_i32_e32 v56, v73
	v_mul_f32_e32 v49, v60, v49
	v_mul_f32_e32 v50, v61, v50
	v_mul_f32_e32 v58, v117, v72
	v_mul_f32_e32 v60, v119, v72
	v_add_f32_e32 v177, v177, v54
	v_add_f32_e32 v174, v174, v53
	v_mul_f32_e32 v53, v114, v72
	v_fmac_f32_e32 v49, v58, v56
	v_fmac_f32_e32 v50, v60, v56
	v_cvt_f32_i32_e32 v51, v51
	v_mul_f32_e32 v54, v112, v72
	v_cvt_f32_i32_e32 v52, v52
	v_add_f32_e32 v172, v172, v49
	v_add_f32_e32 v173, v173, v50
	v_mul_f32_e32 v49, v53, v51
	v_mul_f32_e32 v50, v115, v72
	v_mul_f32_e32 v51, v54, v52
	s_wait_dscnt 0xb
	v_mul_f32_e32 v52, v86, v100
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v53, v113, v72
	v_fmac_f32_e32 v49, v50, v56
	s_wait_dscnt 0xa
	v_mul_f32_e32 v50, v86, v102
	v_cvt_f32_i32_e32 v46, v46
	v_mul_f32_e32 v45, v52, v45
	v_mul_f32_e32 v52, v86, v101
	v_fmac_f32_e32 v51, v53, v56
	v_add_f32_e32 v170, v170, v49
	v_mul_f32_e32 v46, v50, v46
	v_mul_f32_e32 v49, v86, v103
	v_fmac_f32_e32 v45, v52, v87
	s_wait_dscnt 0x9
	v_mul_f32_e32 v50, v86, v96
	s_wait_dscnt 0x8
	v_mul_f32_e32 v52, v86, v98
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_add_f32_e32 v171, v171, v51
	v_fmac_f32_e32 v46, v49, v87
	v_add_f32_e32 v168, v168, v45
	v_mul_f32_e32 v45, v50, v47
	v_mul_f32_e32 v47, v52, v48
	v_mul_f32_e32 v48, v86, v97
	v_mul_f32_e32 v50, v84, v100
	v_mul_f32_e32 v51, v84, v102
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_mul_f32_e32 v49, v86, v99
	v_add_f32_e32 v169, v169, v46
	v_fmac_f32_e32 v45, v48, v87
	v_mul_f32_e32 v41, v50, v41
	v_mul_f32_e32 v42, v51, v42
	v_mul_f32_e32 v46, v84, v101
	v_mul_f32_e32 v48, v84, v103
	v_fmac_f32_e32 v47, v49, v87
	v_mul_f32_e32 v49, v84, v96
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v46, v57
	v_fmac_f32_e32 v42, v48, v57
	v_add_f32_e32 v166, v166, v45
	v_mul_f32_e32 v45, v84, v97
	v_mul_f32_e32 v43, v49, v43
	v_add_f32_e32 v164, v164, v41
	v_add_f32_e32 v165, v165, v42
	v_mul_f32_e32 v41, v82, v100
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v42, v82, v102
	v_cvt_f32_i32_e32 v38, v38
	v_fmac_f32_e32 v43, v45, v57
	v_cvt_f32_i32_e32 v39, v39
	v_mul_f32_e32 v37, v41, v37
	v_mul_f32_e32 v41, v82, v101
	v_mul_f32_e32 v38, v42, v38
	v_mul_f32_e32 v42, v82, v96
	v_add_f32_e32 v162, v162, v43
	v_mul_f32_e32 v43, v82, v103
	v_fmac_f32_e32 v37, v41, v59
	v_mul_f32_e32 v41, v82, v98
	v_cvt_f32_i32_e32 v40, v40
	v_mul_f32_e32 v39, v42, v39
	v_mul_f32_e32 v42, v82, v97
	v_fmac_f32_e32 v38, v43, v59
	v_add_f32_e32 v160, v160, v37
	v_mul_f32_e32 v37, v41, v40
	v_mul_f32_e32 v40, v82, v99
	v_fmac_f32_e32 v39, v42, v59
	v_mul_f32_e32 v41, v72, v100
	v_mul_f32_e32 v42, v72, v102
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v161, v161, v38
	v_fmac_f32_e32 v37, v40, v59
	v_add_f32_e32 v158, v158, v39
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v38, v72, v101
	v_mul_f32_e32 v39, v72, v103
	v_mul_f32_e32 v40, v72, v96
	v_cvt_f32_i32_e32 v35, v35
	v_add_f32_e32 v159, v159, v37
	v_fmac_f32_e32 v33, v38, v56
	v_fmac_f32_e32 v34, v39, v56
	v_mul_f32_e32 v37, v72, v97
	v_mul_f32_e32 v35, v40, v35
	s_wait_dscnt 0x7
	v_mul_f32_e32 v39, v86, v94
	s_wait_dscnt 0x6
	v_mul_f32_e32 v40, v86, v92
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_add_f32_e32 v156, v156, v33
	v_fmac_f32_e32 v35, v37, v56
	v_mul_f32_e32 v33, v86, v95
	v_mul_f32_e32 v29, v39, v29
	v_mul_f32_e32 v30, v40, v30
	v_mul_f32_e32 v37, v86, v93
	v_add_f32_e32 v157, v157, v34
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v29, v33, v87
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v86, v90
	v_fmac_f32_e32 v30, v37, v87
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v86, v88
	v_cvt_f32_i32_e32 v32, v32
	v_add_f32_e32 v152, v152, v29
	v_mul_f32_e32 v29, v33, v31
	v_add_f32_e32 v153, v153, v30
	v_mul_f32_e32 v30, v86, v91
	v_mul_f32_e32 v31, v34, v32
	v_mul_f32_e32 v32, v84, v94
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v33, v86, v89
	v_fmac_f32_e32 v29, v30, v87
	v_mul_f32_e32 v30, v84, v92
	v_cvt_f32_i32_e32 v26, v26
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v32, v84, v95
	v_fmac_f32_e32 v31, v33, v87
	v_add_f32_e32 v150, v150, v29
	v_mul_f32_e32 v26, v30, v26
	v_mul_f32_e32 v29, v84, v93
	v_fmac_f32_e32 v25, v32, v57
	v_mul_f32_e32 v30, v84, v90
	v_mul_f32_e32 v32, v84, v88
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_add_f32_e32 v151, v151, v31
	v_fmac_f32_e32 v26, v29, v57
	v_add_f32_e32 v148, v148, v25
	v_mul_f32_e32 v25, v30, v27
	v_mul_f32_e32 v27, v32, v28
	v_mul_f32_e32 v28, v84, v91
	v_mul_f32_e32 v30, v82, v94
	v_mul_f32_e32 v31, v82, v92
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v29, v84, v89
	v_add_f32_e32 v149, v149, v26
	v_fmac_f32_e32 v25, v28, v57
	v_mul_f32_e32 v21, v30, v21
	v_mul_f32_e32 v22, v31, v22
	v_mul_f32_e32 v26, v82, v95
	v_mul_f32_e32 v28, v82, v93
	v_fmac_f32_e32 v27, v29, v57
	v_mul_f32_e32 v29, v82, v90
	v_cvt_f32_i32_e32 v23, v23
	v_fmac_f32_e32 v21, v26, v59
	v_fmac_f32_e32 v22, v28, v59
	v_add_f32_e32 v146, v146, v25
	v_mul_f32_e32 v25, v82, v91
	v_mul_f32_e32 v23, v29, v23
	v_add_f32_e32 v144, v144, v21
	v_add_f32_e32 v145, v145, v22
	v_mul_f32_e32 v21, v72, v94
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v72, v92
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v23, v25, v59
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v72, v95
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v72, v90
	v_add_f32_e32 v142, v142, v23
	v_mul_f32_e32 v23, v72, v93
	v_fmac_f32_e32 v17, v21, v56
	v_mul_f32_e32 v21, v72, v88
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v72, v91
	v_fmac_f32_e32 v18, v23, v56
	v_add_f32_e32 v140, v140, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v72, v89
	v_fmac_f32_e32 v19, v22, v56
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v86, v80
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v86, v78
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v141, v141, v18
	v_fmac_f32_e32 v17, v20, v56
	v_add_f32_e32 v138, v138, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v86, v81
	v_mul_f32_e32 v19, v86, v79
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v86, v74
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v139, v139, v17
	v_fmac_f32_e32 v13, v18, v87
	v_fmac_f32_e32 v14, v19, v87
	v_mul_f32_e32 v17, v86, v75
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v84, v80
	v_mul_f32_e32 v20, v84, v78
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v136, v136, v13
	v_fmac_f32_e32 v15, v17, v87
	v_mul_f32_e32 v13, v84, v81
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v84, v79
	v_add_f32_e32 v137, v137, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v57
	v_mul_f32_e32 v13, v84, v74
	v_fmac_f32_e32 v10, v17, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v84, v76
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v132, v132, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v133, v133, v10
	v_mul_f32_e32 v10, v84, v75
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v82, v80
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v57
	v_mul_f32_e32 v10, v82, v78
	v_mul_f32_e32 v13, v84, v77
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v82, v81
	v_add_f32_e32 v130, v130, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v82, v79
	v_mul_f32_e32 v10, v82, v74
	v_fmac_f32_e32 v5, v12, v59
	v_mul_f32_e32 v12, v82, v76
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v57
	v_fmac_f32_e32 v6, v9, v59
	v_add_f32_e32 v128, v128, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v82, v75
	v_mul_f32_e32 v9, v82, v77
	v_mul_f32_e32 v105, v116, v84
	v_mul_f32_e32 v50, v84, v98
	v_cvt_f32_i32_e32 v44, v44
	v_mul_f32_e32 v41, v72, v98
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v30, v82, v88
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v21, v86, v76
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v131, v131, v11
	v_mul_f32_e32 v10, v72, v80
	v_mul_f32_e32 v11, v72, v78
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v59
	v_fmac_f32_e32 v7, v9, v59
	v_mul_f32_e32 v8, v72, v74
	v_mul_f32_e32 v9, v72, v76
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v64, v105, v107
	v_mul_f32_e32 v85, v117, v84
	v_mul_f32_e32 v44, v50, v44
	v_mul_f32_e32 v46, v84, v99
	v_mul_f32_e32 v36, v41, v36
	v_mul_f32_e32 v38, v72, v99
	v_mul_f32_e32 v24, v30, v24
	v_mul_f32_e32 v26, v82, v89
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v86, v77
	v_add_f32_e32 v129, v129, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v72, v81
	v_mul_f32_e32 v10, v72, v79
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v72, v75
	v_mul_f32_e32 v9, v72, v77
	v_fmac_f32_e32 v64, v85, v57
	v_fmac_f32_e32 v44, v46, v57
	v_fmac_f32_e32 v36, v38, v56
	v_fmac_f32_e32 v24, v26, v59
	v_fmac_f32_e32 v16, v18, v87
	v_fmac_f32_e32 v1, v6, v56
	v_fmac_f32_e32 v2, v10, v56
	v_fmac_f32_e32 v3, v8, v56
	v_fmac_f32_e32 v4, v9, v56
	v_add_f32_e32 v181, v181, v64
	v_add_f32_e32 v175, v175, v55
	v_add_f32_e32 v167, v167, v47
	v_add_f32_e32 v163, v163, v44
	v_add_f32_e32 v154, v154, v35
	v_add_f32_e32 v155, v155, v36
	v_add_f32_e32 v147, v147, v27
	v_add_f32_e32 v143, v143, v24
	v_add_f32_e32 v134, v134, v15
	v_add_f32_e32 v135, v135, v16
	v_add_f32_e32 v127, v127, v5
	v_add_f32_e32 v126, v126, v7
	v_add_f32_e32 v123, v123, v1
	v_add_f32_e32 v125, v125, v2
	v_add_f32_e32 v67, v67, v3
	v_add_f32_e32 v124, v124, v4
	s_xor_b64 s[34:35], s[30:31], -1
	s_mov_b32 s38, 1
	s_mov_b64 s[30:31], 0
	s_and_b64 vcc, exec, s[34:35]
	s_mov_b64 s[34:35], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_416
.LBB1_419:                              ;   Parent Loop BB1_417 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s22, s38, s25
	s_lshl_b32 s40, s38, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[22:23], s[10:11]
	s_mov_b32 s41, s23
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[42:43], 0x48
	s_add_nc_u64 s[40:41], s[28:29], s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[42:43], s[14:15], s[42:43]
	v_add_nc_u32_e32 v78, s36, v190
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[44:45], s42, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s43, 0, s[44:45]
	v_add_co_u32 v3, s[44:45], s40, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s41, 0, s[44:45]
	v_add_co_u32 v5, s[44:45], s42, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s43, 0, s[44:45]
	v_add_co_u32 v7, s[42:43], s40, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s41, 0, s[42:43]
	global_load_b64 v[110:111], v[1:2], off offset:40
	global_load_b64 v[108:109], v[3:4], off offset:40
	global_load_b64 v[106:107], v[5:6], off offset:40
	global_load_b64 v[104:105], v[7:8], off offset:40
	v_add_nc_u32_e32 v76, s37, v189
	ds_load_2addr_stride64_b32 v[1:2], v78 offset1:2
	ds_load_2addr_stride64_b32 v[3:4], v76 offset1:2
	ds_load_2addr_stride64_b32 v[72:73], v76 offset0:4 offset1:6
	ds_load_2addr_stride64_b32 v[74:75], v78 offset0:4 offset1:6
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v1, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v1, v4, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v1, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v1, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v2, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v2, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v2, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v2, v73, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v74, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v74, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v74, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v74, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v75, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v75, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v75, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v75, v73, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v78 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[74:75], v76 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[76:77], v76 offset0:5 offset1:7
	ds_load_2addr_stride64_b32 v[78:79], v78 offset0:5 offset1:7
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v74, 0x2000, v183
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v110, s[0:1]
	v_add_nc_u32_e32 v73, 0x4000, v183
	v_cndmask_b32_e64 v75, 0, v111, s[0:1]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v74, v108, v109 offset1:16
	ds_store_2addr_b32 v73, v72, v75 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v106, s[2:3]
	v_cndmask_b32_e64 v73, 0, v107, s[2:3]
	v_add_nc_u32_e32 v74, 0x4000, v184
	v_add_nc_u32_e32 v75, 0x2000, v184
	ds_store_2addr_b32 v74, v72, v73 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v75, v104, v105 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[36:37], s[34:35], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_421
; %bb.420:                              ;   in Loop: Header=BB1_419 Depth=2
	s_add_co_i32 s22, s22, 1
	s_xor_b32 s46, s38, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[22:23], s[10:11]
	s_add_co_i32 s22, s38, s24
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[40:41], 0x48
	s_mul_u64 s[42:43], s[22:23], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[40:41], s[14:15], s[40:41]
	s_add_nc_u64 s[38:39], s[12:13], s[42:43]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v122, s[40:41]
	v_add_co_u32 v72, s[44:45], s40, v68
	s_lshl_b32 s42, s46, 6
	s_mov_b32 s43, s23
	s_mulk_i32 s22, 0x88
	v_add_co_ci_u32_e64 v73, null, s41, 0, s[44:45]
	v_add_co_u32 v76, vcc, v76, v194
	v_add_co_u32 v74, s[44:45], s40, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[38:39], s[42:43]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc
	v_add_co_u32 v82, vcc, v65, s22
	v_add_co_ci_u32_e64 v75, null, s41, 0, s[44:45]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s[40:41], s38, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s39, 0, s[40:41]
	v_add_co_u32 v80, s[40:41], s38, v71
	s_lshl_b32 s22, s46, 2
	v_add_co_ci_u32_e64 v81, null, s39, 0, s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc, v82, s22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc
	s_clause 0x1
	global_load_b64 v[110:111], v[72:73], off offset:8
	global_load_b64 v[106:107], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[108:109], v[78:79], off offset:8
	global_load_b64 v[104:105], v[80:81], off offset:8
	global_load_b32 v195, v[76:77], off
	global_load_b32 v196, v[82:83], off
.LBB1_421:                              ; %.preheader527.i.i
                                        ;   in Loop: Header=BB1_419 Depth=2
	v_add_nc_u32_e32 v80, 0, v190
	v_add_nc_u32_e32 v81, 0, v189
	s_xor_b64 s[36:37], s[36:37], -1
	s_and_b64 s[38:39], s[30:31], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:68 offset1:70
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:36 offset1:38
	s_cselect_b32 s38, s21, 0x3c00
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:69 offset1:71
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:37 offset1:39
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s38, v192
	v_add_nc_u32_e32 v72, s22, v193
	s_and_not1_b64 vcc, exec, s[36:37]
	s_movk_i32 s36, 0x2000
	s_movk_i32 s37, 0x4000
	ds_load_2addr_b32 v[116:117], v76 offset1:1
	ds_load_2addr_b32 v[118:119], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[112:113], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[86:87], v72 offset1:1
	ds_load_2addr_b32 v[84:85], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[82:83], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[100:101], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[102:103], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[96:97], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[98:99], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[94:95], v76 offset0:64 offset1:65
	ds_load_2addr_b32 v[92:93], v76 offset0:66 offset1:67
	ds_load_2addr_b32 v[90:91], v76 offset0:68 offset1:69
	ds_load_2addr_b32 v[88:89], v76 offset0:70 offset1:71
	ds_load_2addr_b32 v[80:81], v76 offset0:96 offset1:97
	ds_load_2addr_b32 v[78:79], v76 offset0:98 offset1:99
	ds_load_2addr_b32 v[74:75], v76 offset0:100 offset1:101
	ds_load_2addr_b32 v[76:77], v76 offset0:102 offset1:103
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_418
; %bb.422:                              ; %.preheader528.i.i
                                        ;   in Loop: Header=BB1_419 Depth=2
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v110, 0, v110, s[0:1]
	v_cndmask_b32_e64 v111, 0, v111, s[0:1]
	v_add_nc_u32_e32 v197, 0x1000, v183
	s_and_b64 s[34:35], s[34:35], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_store_2addr_b32 v183, v110, v111 offset1:16
	s_wait_loadcnt 0x3
	ds_store_2addr_b32 v197, v108, v109 offset1:16
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v108, v191, v196
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v111, s22, v121
	s_cselect_b32 s22, s21, 0x3c00
	v_cndmask_b32_e64 v106, 0, v106, s[2:3]
	v_cndmask_b32_e64 v107, 0, v107, s[2:3]
	v_cvt_f32_f16_e32 v108, v108.l
	v_cndmask_b32_e64 v110, 0, v195, s[4:5]
	v_add_nc_u32_e32 v109, 0x1000, v184
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v197, s22, v121
	s_mov_b32 s37, 0
	v_cndmask_b32_e64 v108, 0, v108, s[6:7]
	s_movk_i32 s36, 0x1000
	ds_store_2addr_b32 v184, v106, v107 offset1:16
	ds_store_2addr_b32 v109, v104, v105 offset1:16
	ds_store_b32 v111, v110
	ds_store_b32 v197, v108
	s_branch .LBB1_418
.LBB1_423:                              ; %._crit_edge599.i.i
	v_and_b32_e32 v3, 15, v0
	v_mul_u32_u24_e32 v1, 0x500, v120
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s9, s8, 31
	s_ashr_i32 s21, s20, 31
	v_lshlrev_b32_e32 v2, 2, v3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[8:9], s[18:19]
	s_lshl_b64 s[2:3], s[20:21], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_mov_b32 s7, 0x31004000
	v_add3_u32 v6, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	s_mov_b32 s6, -1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	s_add_co_i32 s0, s20, 0x80
	v_mad_i32_i24 v1, 0x50, v178, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s0, s8
	ds_store_2addr_b32 v1, v187, v188 offset1:20
	ds_store_2addr_b32 v1, v185, v186 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshrrev_b32_e32 v1, 4, v0
	v_mul_u32_u24_e32 v0, 0x50, v3
	s_cselect_b64 s[0:1], -1, 0
	s_add_co_i32 s2, s18, 0x80
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 2, v1
	v_mul_lo_u32 v7, s8, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s2, s10
	v_or_b32_e32 v4, s18, v1
	s_cselect_b64 s[2:3], -1, 0
	v_add3_u32 v2, 0, v0, v2
	v_or_b32_e32 v0, s20, v3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[2:3], s[0:1], s[2:3]
	s_mov_b64 s[0:1], -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[2:3]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v5, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_427
; %bb.424:
	v_cmp_gt_i32_e32 vcc, s8, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_and_b64 s[12:13], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[12:13]
	s_cbranch_execz .LBB1_426
; %bb.425:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off
.LBB1_426:                              ; %Flow2405
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_427:                              ; %Flow2406
	v_add_lshl_u32 v3, v7, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_429
; %bb.428:
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], null offen
.LBB1_429:
	ds_load_b32 v7, v2 offset:1280
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v5, 0, 1, s[2:3]
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mov_b64 s[0:1], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_433
; %bb.430:
	v_or_b32_e32 v1, 64, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_432
; %bb.431:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_432:                              ; %Flow2403
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_433:                              ; %Flow2404
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_435
; %bb.434:
	s_lshl_b32 s0, s8, 8
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_435:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_439
; %bb.436:
	v_or_b32_e32 v1, 64, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_438
; %bb.437:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_438:                              ; %Flow2401
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_439:                              ; %Flow2402
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_441
; %bb.440:
	s_movk_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_441:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_445
; %bb.442:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_444
; %bb.443:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_444:                              ; %Flow2399
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_445:                              ; %Flow2400
	v_mul_i32_i24_e32 v1, 0x50, v178
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_447
; %bb.446:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_447:                              ; %.preheader.1.i.i39
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v6, v6, v1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v181, v182 offset1:20
	ds_store_2addr_b32 v6, v179, v180 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_451
; %bb.448:
	v_or_b32_e32 v1, 16, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_450
; %bb.449:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_450:                              ; %Flow2397
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_451:                              ; %Flow2398
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_453
; %bb.452:
	s_lshl_b32 s0, s8, 6
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_453:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_457
; %bb.454:
	v_or_b32_e32 v1, 0x50, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_456
; %bb.455:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_456:                              ; %Flow2395
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_457:                              ; %Flow2396
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_459
; %bb.458:
	s_mul_i32 s0, s8, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_459:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_463
; %bb.460:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_462
; %bb.461:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_462:                              ; %Flow2393
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_463:                              ; %Flow2394
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_465
; %bb.464:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_465:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_469
; %bb.466:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_468
; %bb.467:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_468:                              ; %Flow2391
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_469:                              ; %Flow2392
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_471
; %bb.470:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_471:                              ; %.preheader.2.i.i40
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v176, v177 offset1:20
	ds_store_2addr_b32 v6, v174, v175 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_475
; %bb.472:
	v_or_b32_e32 v1, 32, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_474
; %bb.473:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_474:                              ; %Flow2389
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_475:                              ; %Flow2390
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_477
; %bb.476:
	s_lshl_b32 s0, s8, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_477:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_481
; %bb.478:
	v_or_b32_e32 v1, 0x60, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_480
; %bb.479:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_480:                              ; %Flow2387
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_481:                              ; %Flow2388
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_483
; %bb.482:
	s_mul_i32 s0, s8, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_483:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_487
; %bb.484:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_486
; %bb.485:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_486:                              ; %Flow2385
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_487:                              ; %Flow2386
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_489
; %bb.488:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_489:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_493
; %bb.490:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_492
; %bb.491:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_492:                              ; %Flow2383
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_493:                              ; %Flow2384
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_495
; %bb.494:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_495:                              ; %.preheader.3.i.i41
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v172, v173 offset1:20
	ds_store_2addr_b32 v6, v170, v171 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_499
; %bb.496:
	v_or_b32_e32 v1, 48, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_498
; %bb.497:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_498:                              ; %Flow2381
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_499:                              ; %Flow2382
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_501
; %bb.500:
	s_mul_i32 s0, s8, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_501:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_505
; %bb.502:
	v_or_b32_e32 v1, 0x70, v4
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_504
; %bb.503:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_504:                              ; %Flow2379
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_505:                              ; %Flow2380
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_507
; %bb.506:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_507:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_511
; %bb.508:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_510
; %bb.509:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_510:                              ; %Flow2377
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_511:                              ; %Flow2378
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_513
; %bb.512:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_513:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_517
; %bb.514:
	v_or_b32_e32 v8, 64, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_516
; %bb.515:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:256
.LBB1_516:                              ; %Flow2375
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_517:                              ; %Flow2376
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_519
; %bb.518:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_519:                              ; %.preheader524.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v168, v169 offset1:20
	ds_store_2addr_b32 v6, v166, v167 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_523
; %bb.520:
	v_or_b32_e32 v1, 16, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_522
; %bb.521:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_522:                              ; %Flow2373
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_523:                              ; %Flow2374
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_525
; %bb.524:
	s_mov_b32 s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_525:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_529
; %bb.526:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_528
; %bb.527:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_528:                              ; %Flow2371
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_529:                              ; %Flow2372
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_531
; %bb.530:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_531:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_535
; %bb.532:
	v_or_b32_e32 v1, 0x50, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_534
; %bb.533:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_534:                              ; %Flow2369
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_535:                              ; %Flow2370
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_537
; %bb.536:
	s_movk_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_537:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_541
; %bb.538:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_540
; %bb.539:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_540:                              ; %Flow2367
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_541:                              ; %Flow2368
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_543
; %bb.542:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_543:                              ; %.preheader.1.1.i.i42
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v164, v165 offset1:20
	ds_store_2addr_b32 v6, v162, v163 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_547
; %bb.544:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_546
; %bb.545:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_546:                              ; %Flow2365
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_547:                              ; %Flow2366
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_549
; %bb.548:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_549:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_553
; %bb.550:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_552
; %bb.551:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_552:                              ; %Flow2363
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_553:                              ; %Flow2364
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_555
; %bb.554:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_555:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_559
; %bb.556:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_558
; %bb.557:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_558:                              ; %Flow2361
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_559:                              ; %Flow2362
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_561
; %bb.560:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_561:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_565
; %bb.562:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_564
; %bb.563:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_564:                              ; %Flow2359
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_565:                              ; %Flow2360
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_567
; %bb.566:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_567:                              ; %.preheader.2.1.i.i43
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v160, v161 offset1:20
	ds_store_2addr_b32 v6, v158, v159 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_571
; %bb.568:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_570
; %bb.569:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_570:                              ; %Flow2357
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_571:                              ; %Flow2358
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_573
; %bb.572:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_573:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_577
; %bb.574:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_576
; %bb.575:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_576:                              ; %Flow2355
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_577:                              ; %Flow2356
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_579
; %bb.578:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_579:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_583
; %bb.580:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_582
; %bb.581:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_582:                              ; %Flow2353
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_583:                              ; %Flow2354
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_585
; %bb.584:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_585:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_589
; %bb.586:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_588
; %bb.587:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_588:                              ; %Flow2351
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_589:                              ; %Flow2352
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_591
; %bb.590:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_591:                              ; %.preheader.3.1.i.i44
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v156, v157 offset1:20
	ds_store_2addr_b32 v6, v154, v155 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_595
; %bb.592:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_594
; %bb.593:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_594:                              ; %Flow2349
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_595:                              ; %Flow2350
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_597
; %bb.596:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_597:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_601
; %bb.598:
	v_or_b32_e32 v8, 16, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_600
; %bb.599:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:64
.LBB1_600:                              ; %Flow2347
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_601:                              ; %Flow2348
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_603
; %bb.602:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_603:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_607
; %bb.604:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_606
; %bb.605:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_606:                              ; %Flow2345
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_607:                              ; %Flow2346
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_609
; %bb.608:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_609:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_613
; %bb.610:
	v_or_b32_e32 v8, 0x50, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_612
; %bb.611:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:320
.LBB1_612:                              ; %Flow2343
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_613:                              ; %Flow2344
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_615
; %bb.614:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_615:                              ; %.preheader524.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v152, v153 offset1:20
	ds_store_2addr_b32 v6, v150, v151 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_619
; %bb.616:
	v_or_b32_e32 v1, 32, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_618
; %bb.617:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_618:                              ; %Flow2341
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_619:                              ; %Flow2342
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_621
; %bb.620:
	s_movk_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_621:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_625
; %bb.622:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_624
; %bb.623:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_624:                              ; %Flow2339
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_625:                              ; %Flow2340
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_627
; %bb.626:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_bitset1_b32 s0, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_627:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_631
; %bb.628:
	v_or_b32_e32 v1, 0x60, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_630
; %bb.629:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_630:                              ; %Flow2337
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_631:                              ; %Flow2338
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_633
; %bb.632:
	s_movk_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_633:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_637
; %bb.634:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_636
; %bb.635:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_636:                              ; %Flow2335
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_637:                              ; %Flow2336
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_639
; %bb.638:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_639:                              ; %.preheader.1.2.i.i45
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v148, v149 offset1:20
	ds_store_2addr_b32 v6, v146, v147 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_643
; %bb.640:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_642
; %bb.641:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_642:                              ; %Flow2333
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_643:                              ; %Flow2334
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_645
; %bb.644:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_645:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_649
; %bb.646:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_648
; %bb.647:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_648:                              ; %Flow2331
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_649:                              ; %Flow2332
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_651
; %bb.650:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_651:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_655
; %bb.652:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_654
; %bb.653:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_654:                              ; %Flow2329
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_655:                              ; %Flow2330
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_657
; %bb.656:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_657:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_661
; %bb.658:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_660
; %bb.659:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_660:                              ; %Flow2327
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_661:                              ; %Flow2328
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_663
; %bb.662:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_663:                              ; %.preheader.2.2.i.i46
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v144, v145 offset1:20
	ds_store_2addr_b32 v6, v142, v143 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_667
; %bb.664:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_666
; %bb.665:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_666:                              ; %Flow2325
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_667:                              ; %Flow2326
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_669
; %bb.668:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_669:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_673
; %bb.670:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_672
; %bb.671:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_672:                              ; %Flow2323
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_673:                              ; %Flow2324
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_675
; %bb.674:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_675:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_679
; %bb.676:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_678
; %bb.677:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_678:                              ; %Flow2321
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_679:                              ; %Flow2322
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_681
; %bb.680:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_681:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_685
; %bb.682:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_684
; %bb.683:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_684:                              ; %Flow2319
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_685:                              ; %Flow2320
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_687
; %bb.686:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_687:                              ; %.preheader.3.2.i.i47
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v140, v141 offset1:20
	ds_store_2addr_b32 v6, v138, v139 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_691
; %bb.688:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_690
; %bb.689:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_690:                              ; %Flow2317
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_691:                              ; %Flow2318
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_693
; %bb.692:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_693:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_697
; %bb.694:
	v_or_b32_e32 v8, 32, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_696
; %bb.695:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:128
.LBB1_696:                              ; %Flow2315
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_697:                              ; %Flow2316
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_699
; %bb.698:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_699:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_703
; %bb.700:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_702
; %bb.701:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_702:                              ; %Flow2313
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_703:                              ; %Flow2314
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_705
; %bb.704:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_705:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_709
; %bb.706:
	v_or_b32_e32 v8, 0x60, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_708
; %bb.707:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:384
.LBB1_708:                              ; %Flow2311
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_709:                              ; %Flow2312
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_711
; %bb.710:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_711:                              ; %.preheader524.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v136, v137 offset1:20
	ds_store_2addr_b32 v6, v134, v135 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_715
; %bb.712:
	v_or_b32_e32 v1, 48, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_714
; %bb.713:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_714:                              ; %Flow2309
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_715:                              ; %Flow2310
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_717
; %bb.716:
	s_movk_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_717:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_721
; %bb.718:
	v_or_b32_e32 v8, 48, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_720
; %bb.719:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_720:                              ; %Flow2307
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_721:                              ; %Flow2308
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_723
; %bb.722:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_723:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_727
; %bb.724:
	v_or_b32_e32 v1, 0x70, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_726
; %bb.725:
	v_mad_co_i64_i32 v[8:9], null, s8, v4, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_726:                              ; %Flow2305
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_727:                              ; %Flow2306
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_729
; %bb.728:
	s_movk_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_729:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_733
; %bb.730:
	v_or_b32_e32 v8, 0x70, v0
	v_or_b32_e32 v1, 64, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_732
; %bb.731:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_732:                              ; %Flow2303
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_733:                              ; %Flow2304
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_735
; %bb.734:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_735:                              ; %.preheader.1.3.i.i48
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v132, v133 offset1:20
	ds_store_2addr_b32 v6, v130, v131 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_739
; %bb.736:
	v_or_b32_e32 v8, 48, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_738
; %bb.737:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_738:                              ; %Flow2301
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_739:                              ; %Flow2302
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_741
; %bb.740:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_741:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_745
; %bb.742:
	v_or_b32_e32 v8, 48, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_744
; %bb.743:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_744:                              ; %Flow2299
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_745:                              ; %Flow2300
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_747
; %bb.746:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_747:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_751
; %bb.748:
	v_or_b32_e32 v8, 0x70, v0
	v_or_b32_e32 v1, 16, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_750
; %bb.749:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_750:                              ; %Flow2297
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_751:                              ; %Flow2298
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_753
; %bb.752:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_753:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_757
; %bb.754:
	v_or_b32_e32 v8, 0x70, v0
	v_or_b32_e32 v1, 0x50, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_756
; %bb.755:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_756:                              ; %Flow2295
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_757:                              ; %Flow2296
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_759
; %bb.758:
	s_mul_i32 s0, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_759:                              ; %.preheader.2.3.i.i49
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v128, v129 offset1:20
	ds_store_2addr_b32 v6, v127, v126 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v7, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_763
; %bb.760:
	v_or_b32_e32 v8, 48, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_762
; %bb.761:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_762:                              ; %Flow2293
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_763:                              ; %Flow2294
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_765
; %bb.764:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_765:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_769
; %bb.766:
	v_or_b32_e32 v8, 48, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_768
; %bb.767:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:192
.LBB1_768:                              ; %Flow2291
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_769:                              ; %Flow2292
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_771
; %bb.770:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_771:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_775
; %bb.772:
	v_or_b32_e32 v8, 0x70, v0
	v_or_b32_e32 v1, 32, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_774
; %bb.773:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_774:                              ; %Flow2289
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_775:                              ; %Flow2290
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_777
; %bb.776:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_777:
	s_wait_dscnt 0x0
	ds_load_b32 v7, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_781
; %bb.778:
	v_or_b32_e32 v8, 0x70, v0
	v_or_b32_e32 v1, 0x60, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v8
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_780
; %bb.779:
	v_mad_co_i64_i32 v[8:9], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off offset:448
.LBB1_780:                              ; %Flow2287
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_781:                              ; %Flow2288
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_783
; %bb.782:
	s_mul_i32 s0, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s0 offen
.LBB1_783:                              ; %.preheader.3.3.i.i50
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v123, v125 offset1:20
	ds_store_2addr_b32 v6, v67, v124 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_787
; %bb.784:
	v_or_b32_e32 v7, 48, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v7
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_786
; %bb.785:
	v_mad_co_i64_i32 v[7:8], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:192
.LBB1_786:                              ; %Flow2285
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_787:                              ; %Flow2286
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_789
; %bb.788:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB1_789:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_793
; %bb.790:
	v_or_b32_e32 v7, 48, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v7
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_792
; %bb.791:
	v_mad_co_i64_i32 v[7:8], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:192
.LBB1_792:                              ; %Flow2283
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_793:                              ; %Flow2284
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_795
; %bb.794:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB1_795:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_799
; %bb.796:
	v_or_b32_e32 v7, 0x70, v0
	v_or_b32_e32 v1, 48, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v7
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_798
; %bb.797:
	v_mad_co_i64_i32 v[7:8], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:448
.LBB1_798:                              ; %Flow2281
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_799:                              ; %Flow2282
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_801
; %bb.800:
	s_mul_i32 s0, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB1_801:
	ds_load_b32 v2, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v5
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB1_805
; %bb.802:
	v_or_b32_e32 v5, 0x70, v0
	v_or_b32_e32 v1, 0x70, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc, s8, v5
	v_cmp_gt_i32_e64 s[0:1], s10, v1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB1_804
; %bb.803:
	v_mad_co_i64_i32 v[4:5], null, s8, v1, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v2, off offset:448
.LBB1_804:                              ; %Flow2278
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_805:                              ; %Flow2280
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_807
; %bb.806:
	s_mul_i32 s0, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v3, s[4:7], s0 offen
.LBB1_807:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b64 s[0:1], -1
	s_barrier_wait -1
.LBB1_808:                              ; %Flow2416
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_810
; %bb.809:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_810:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.Lfunc_end1:
	.size	iu4_w64_r4, .Lfunc_end1-iu4_w64_r4
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r4
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
		.amdhsa_next_free_vgpr 200
		.amdhsa_next_free_sgpr 52
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-iu4_w64_r4)<<4)&4080)>>4
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
	.set .Liu4_w64_r4.num_vgpr, 200
	.set .Liu4_w64_r4.num_agpr, 0
	.set .Liu4_w64_r4.numbered_sgpr, 52
	.set .Liu4_w64_r4.num_named_barrier, 0
	.set .Liu4_w64_r4.private_seg_size, 0
	.set .Liu4_w64_r4.uses_vcc, 1
	.set .Liu4_w64_r4.uses_flat_scratch, 0
	.set .Liu4_w64_r4.has_dyn_sized_stack, 0
	.set .Liu4_w64_r4.has_recursion, 0
	.set .Liu4_w64_r4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 41052
; TotalNumSgprs: 54
; NumVgprs: 200
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 49
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 200
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	iu4_w64_r4_add          ; -- Begin function iu4_w64_r4_add
	.globl	iu4_w64_r4_add
	.p2align	8
	.type	iu4_w64_r4_add,@function
iu4_w64_r4_add:                         ; @iu4_w64_r4_add
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[8:10], s[0:1], 0x18
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s20, s8
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_ge_i32 s18, s10
	s_cselect_b64 s[4:5], -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB2_403
; %bb.1:                                ; %.preheader538.i.i
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s9, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s9, s0
	s_add_co_i32 s7, s8, -1
	s_ashr_i32 s19, s0, 8
	s_mov_b64 s[0:1], exec
	s_mul_i32 s6, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i.i
	v_or_b32_e32 v5, s18, v2
	v_mov_b32_e32 v4, 0
	v_lshlrev_b32_e32 v3, 2, v1
	s_mov_b64 s[2:3], exec
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s10, v5
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc
	global_load_b32 v4, v[4:5], off
.LBB2_4:                                ; %.preheader535.loopexit.i.i
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v7, s20, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s7, v7
	v_cmp_gt_i32_e32 vcc, s8, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s6, v5, s[12:13]
	global_load_b32 v5, v[5:6], off
	v_lshlrev_b32_e32 v6, 4, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v5, v6, v5
	v_lshlrev_b32_e32 v6, 3, v2
	v_cvt_f32_f16_e32 v5, v5.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v3, 0, v6, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, 0, v5, vcc
	ds_store_2addr_stride64_b32 v3, v4, v5 offset0:48 offset1:56
.LBB2_5:                                ; %Flow1137
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v3, 2, v0
	s_add_co_i32 s21, s10, -1
	v_lshrrev_b32_e32 v13, 5, v0
	v_lshlrev_b32_e32 v14, 7, v0
	v_and_b32_e32 v15, 60, v0
	v_add_nc_u32_e32 v11, s18, v3
	v_add_nc_u32_e32 v4, s20, v3
	v_and_b32_e32 v3, 3, v0
	v_bfe_u32 v16, v0, 1, 1
	v_or_b32_e32 v17, 8, v13
	v_add_nc_u32_e32 v12, 64, v11
	v_add_nc_u32_e32 v5, 64, v4
	v_lshlrev_b32_e32 v3, 3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v6, s21, v11
	v_min_i32_e32 v4, s7, v4
	v_min_i32_e32 v7, s21, v12
	v_min_i32_e32 v5, s7, v5
	v_and_or_b32 v14, 0x80, v14, v15
	v_and_or_b32 v13, v13, 6, v16
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s6, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s6, v5, v[3:4]
	v_and_or_b32 v15, v17, 14, v16
	v_lshl_or_b32 v13, v13, 8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[14:15] offset:8
	global_load_b64 v[5:6], v69, s[12:13] offset:8
	global_load_b64 v[7:8], v70, s[14:15] offset:8
	global_load_b64 v[9:10], v71, s[12:13] offset:8
	v_lshl_or_b32 v14, v15, 8, v14
	v_add_nc_u32_e32 v182, 0, v13
	v_cmp_gt_i32_e64 s[2:3], s10, v12
	s_cmp_gt_i32 s9, 0xff
	v_add_nc_u32_e32 v185, 0, v14
	v_add_nc_u32_e32 v11, 0x1000, v182
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v12, 0x1000, v185
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v3, 0, v3, s[0:1]
	v_cndmask_b32_e64 v4, 0, v4, s[0:1]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v7, 0, v7, s[2:3]
	v_cndmask_b32_e64 v8, 0, v8, s[2:3]
	ds_store_2addr_b32 v11, v5, v6 offset1:16
	ds_store_2addr_b32 v182, v3, v4 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v12, v9, v10 offset1:16
	ds_store_2addr_b32 v185, v7, v8 offset1:16
	v_bfe_u32 v3, v0, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v179, 2, v3
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_7
; %bb.6:                                ; %._crit_edge.._crit_edge604_crit_edge.i.i
	v_lshlrev_b32_e32 v5, 2, v3
	s_mov_b64 s[4:5], 0
	s_branch .LBB2_8
.LBB2_7:
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr5
.LBB2_8:                                ; %Flow1135
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_18
; %bb.9:                                ; %.preheader534.lr.ph.i.i
	v_add_nc_u32_e32 v3, s20, v2
	v_and_b32_e32 v4, 63, v0
	v_and_b32_e32 v5, 64, v2
	v_add_nc_u32_e32 v8, s18, v2
	v_lshlrev_b32_e32 v2, 3, v2
	v_min_i32_e32 v7, s7, v3
	v_lshlrev_b32_e32 v9, 2, v1
	v_lshlrev_b32_e32 v10, 3, v0
	v_lshlrev_b32_e32 v6, 5, v0
	v_lshlrev_b32_e32 v4, 2, v4
	v_mad_co_u64_u32 v[65:66], null, s6, v7, s[12:13]
	v_ashrrev_i32_e32 v7, 31, v7
	v_or_b32_e32 v11, v179, v5
	v_add3_u32 v189, 0, v2, v9
	v_and_b32_e32 v2, 0x278, v10
	v_mov_b32_e32 v120, 0
	v_min_i32_e32 v186, s21, v8
	v_and_or_b32 v187, 0x800, v6, v4
	v_mad_co_u64_u32 v[66:67], null, s6, v7, v[66:67]
	v_lshl_or_b32 v188, v5, 5, v4
	v_cmp_gt_i32_e64 s[4:5], s10, v8
	v_cmp_gt_i32_e64 s[6:7], s8, v3
	v_lshlrev_b32_e32 v190, 4, v1
	v_lshl_add_u32 v191, v11, 3, 0
	v_add_nc_u32_e32 v192, 0, v2
	v_lshlrev_b32_e32 v193, 2, v1
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	s_mov_b32 s23, 0
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s36, 0x1000
	s_movk_i32 s9, 0x3000
	s_movk_i32 s21, 0x3800
	s_mov_b32 s37, 0
	s_mov_b32 s24, s23
	s_branch .LBB2_11
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=1
	s_and_b64 vcc, exec, s[26:27]
	s_mov_b32 s24, s33
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_17
.LBB2_11:                               ; %.preheader534.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_13 Depth 2
	s_mov_b32 s25, s23
	s_add_co_i32 s33, s24, 1
	s_mul_u64 s[28:29], s[24:25], 0x88
	s_lshl_b32 s25, s24, 1
	s_cmp_eq_u32 s33, s19
	s_mov_b64 s[34:35], 0
	s_cselect_b64 s[26:27], -1, 0
	s_add_nc_u64 s[28:29], s[12:13], s[28:29]
	s_mov_b64 s[30:31], -1
	s_mov_b32 s38, s23
	s_branch .LBB2_13
.LBB2_12:                               ;   in Loop: Header=BB2_13 Depth=2
	s_wait_loadcnt_dscnt 0x20f
	v_mul_f32_e32 v104, v116, v86
	v_cvt_f32_i32_e32 v61, v61
	v_cvt_f32_i32_e32 v87, v87
	v_mul_f32_e32 v105, v118, v86
	v_cvt_f32_i32_e32 v62, v62
	v_mul_f32_e32 v106, v117, v86
	v_mul_f32_e32 v61, v104, v61
	v_mul_f32_e32 v104, v119, v86
	v_mul_f32_e32 v107, v114, v86
	v_mul_f32_e32 v62, v105, v62
	v_mul_f32_e32 v105, v112, v86
	v_fmac_f32_e32 v61, v106, v87
	v_cvt_f32_i32_e32 v63, v63
	v_cvt_f32_i32_e32 v64, v64
	v_fmac_f32_e32 v62, v104, v87
	v_mul_f32_e32 v104, v113, v86
	v_add_f32_e32 v183, v183, v61
	v_mul_f32_e32 v61, v107, v63
	v_mul_f32_e32 v63, v105, v64
	v_mul_f32_e32 v64, v115, v86
	s_wait_dscnt 0xe
	v_mul_f32_e32 v106, v118, v84
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v107, v57
	v_fmac_f32_e32 v63, v104, v87
	v_fmac_f32_e32 v61, v64, v87
	v_cvt_f32_i32_e32 v57, v85
	v_mul_f32_e32 v58, v106, v58
	v_mul_f32_e32 v104, v119, v84
	v_add_f32_e32 v184, v184, v62
	v_add_f32_e32 v180, v180, v61
	v_mul_f32_e32 v61, v114, v84
	v_cvt_f32_i32_e32 v59, v59
	v_fmac_f32_e32 v58, v104, v57
	v_mul_f32_e32 v62, v112, v84
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v53, v53
	v_add_f32_e32 v181, v181, v63
	v_add_f32_e32 v178, v178, v58
	v_mul_f32_e32 v58, v61, v59
	v_mul_f32_e32 v59, v115, v84
	v_mul_f32_e32 v60, v62, v60
	s_wait_dscnt 0xd
	v_mul_f32_e32 v62, v116, v82
	v_mul_f32_e32 v61, v113, v84
	v_mul_f32_e32 v63, v118, v82
	v_fmac_f32_e32 v58, v59, v57
	v_cvt_f32_i32_e32 v59, v83
	v_cvt_f32_i32_e32 v54, v54
	v_mul_f32_e32 v53, v62, v53
	v_mul_f32_e32 v62, v117, v82
	v_fmac_f32_e32 v60, v61, v57
	v_add_f32_e32 v175, v175, v58
	v_mul_f32_e32 v54, v63, v54
	v_mul_f32_e32 v58, v119, v82
	v_fmac_f32_e32 v53, v62, v59
	v_mul_f32_e32 v61, v114, v82
	v_mul_f32_e32 v62, v112, v82
	v_cvt_f32_i32_e32 v55, v55
	v_cvt_f32_i32_e32 v56, v56
	v_add_f32_e32 v176, v176, v60
	v_fmac_f32_e32 v54, v58, v59
	v_add_f32_e32 v173, v173, v53
	v_mul_f32_e32 v53, v61, v55
	v_mul_f32_e32 v55, v62, v56
	v_mul_f32_e32 v56, v115, v82
	v_mul_f32_e32 v58, v113, v82
	s_wait_dscnt 0xc
	v_mul_f32_e32 v60, v116, v72
	v_mul_f32_e32 v61, v118, v72
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_fmac_f32_e32 v53, v56, v59
	v_fmac_f32_e32 v55, v58, v59
	v_cvt_f32_i32_e32 v56, v73
	v_mul_f32_e32 v49, v60, v49
	v_mul_f32_e32 v50, v61, v50
	v_mul_f32_e32 v58, v117, v72
	v_mul_f32_e32 v60, v119, v72
	v_add_f32_e32 v174, v174, v54
	v_add_f32_e32 v171, v171, v53
	v_mul_f32_e32 v53, v114, v72
	v_fmac_f32_e32 v49, v58, v56
	v_fmac_f32_e32 v50, v60, v56
	v_cvt_f32_i32_e32 v51, v51
	v_mul_f32_e32 v54, v112, v72
	v_cvt_f32_i32_e32 v52, v52
	v_add_f32_e32 v169, v169, v49
	v_add_f32_e32 v170, v170, v50
	v_mul_f32_e32 v49, v53, v51
	v_mul_f32_e32 v50, v115, v72
	v_mul_f32_e32 v51, v54, v52
	s_wait_dscnt 0xb
	v_mul_f32_e32 v52, v86, v100
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v53, v113, v72
	v_fmac_f32_e32 v49, v50, v56
	s_wait_dscnt 0xa
	v_mul_f32_e32 v50, v86, v102
	v_cvt_f32_i32_e32 v46, v46
	v_mul_f32_e32 v45, v52, v45
	v_mul_f32_e32 v52, v86, v101
	v_fmac_f32_e32 v51, v53, v56
	v_add_f32_e32 v167, v167, v49
	v_mul_f32_e32 v46, v50, v46
	v_mul_f32_e32 v49, v86, v103
	v_fmac_f32_e32 v45, v52, v87
	s_wait_dscnt 0x9
	v_mul_f32_e32 v50, v86, v96
	s_wait_dscnt 0x8
	v_mul_f32_e32 v52, v86, v98
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_add_f32_e32 v168, v168, v51
	v_fmac_f32_e32 v46, v49, v87
	v_add_f32_e32 v165, v165, v45
	v_mul_f32_e32 v45, v50, v47
	v_mul_f32_e32 v47, v52, v48
	v_mul_f32_e32 v48, v86, v97
	v_mul_f32_e32 v50, v84, v100
	v_mul_f32_e32 v51, v84, v102
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_mul_f32_e32 v49, v86, v99
	v_add_f32_e32 v166, v166, v46
	v_fmac_f32_e32 v45, v48, v87
	v_mul_f32_e32 v41, v50, v41
	v_mul_f32_e32 v42, v51, v42
	v_mul_f32_e32 v46, v84, v101
	v_mul_f32_e32 v48, v84, v103
	v_fmac_f32_e32 v47, v49, v87
	v_mul_f32_e32 v49, v84, v96
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v46, v57
	v_fmac_f32_e32 v42, v48, v57
	v_add_f32_e32 v163, v163, v45
	v_mul_f32_e32 v45, v84, v97
	v_mul_f32_e32 v43, v49, v43
	v_add_f32_e32 v161, v161, v41
	v_add_f32_e32 v162, v162, v42
	v_mul_f32_e32 v41, v82, v100
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v42, v82, v102
	v_cvt_f32_i32_e32 v38, v38
	v_fmac_f32_e32 v43, v45, v57
	v_cvt_f32_i32_e32 v39, v39
	v_mul_f32_e32 v37, v41, v37
	v_mul_f32_e32 v41, v82, v101
	v_mul_f32_e32 v38, v42, v38
	v_mul_f32_e32 v42, v82, v96
	v_add_f32_e32 v159, v159, v43
	v_mul_f32_e32 v43, v82, v103
	v_fmac_f32_e32 v37, v41, v59
	v_mul_f32_e32 v41, v82, v98
	v_cvt_f32_i32_e32 v40, v40
	v_mul_f32_e32 v39, v42, v39
	v_mul_f32_e32 v42, v82, v97
	v_fmac_f32_e32 v38, v43, v59
	v_add_f32_e32 v157, v157, v37
	v_mul_f32_e32 v37, v41, v40
	v_mul_f32_e32 v40, v82, v99
	v_fmac_f32_e32 v39, v42, v59
	v_mul_f32_e32 v41, v72, v100
	v_mul_f32_e32 v42, v72, v102
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v158, v158, v38
	v_fmac_f32_e32 v37, v40, v59
	v_add_f32_e32 v155, v155, v39
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v38, v72, v101
	v_mul_f32_e32 v39, v72, v103
	v_mul_f32_e32 v40, v72, v96
	v_cvt_f32_i32_e32 v35, v35
	v_add_f32_e32 v156, v156, v37
	v_fmac_f32_e32 v33, v38, v56
	v_fmac_f32_e32 v34, v39, v56
	v_mul_f32_e32 v37, v72, v97
	v_mul_f32_e32 v35, v40, v35
	s_wait_dscnt 0x7
	v_mul_f32_e32 v39, v86, v94
	s_wait_dscnt 0x6
	v_mul_f32_e32 v40, v86, v92
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_add_f32_e32 v153, v153, v33
	v_fmac_f32_e32 v35, v37, v56
	v_mul_f32_e32 v33, v86, v95
	v_mul_f32_e32 v29, v39, v29
	v_mul_f32_e32 v30, v40, v30
	v_mul_f32_e32 v37, v86, v93
	v_add_f32_e32 v154, v154, v34
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v29, v33, v87
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v86, v90
	v_fmac_f32_e32 v30, v37, v87
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v86, v88
	v_cvt_f32_i32_e32 v32, v32
	v_add_f32_e32 v149, v149, v29
	v_mul_f32_e32 v29, v33, v31
	v_add_f32_e32 v150, v150, v30
	v_mul_f32_e32 v30, v86, v91
	v_mul_f32_e32 v31, v34, v32
	v_mul_f32_e32 v32, v84, v94
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v33, v86, v89
	v_fmac_f32_e32 v29, v30, v87
	v_mul_f32_e32 v30, v84, v92
	v_cvt_f32_i32_e32 v26, v26
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v32, v84, v95
	v_fmac_f32_e32 v31, v33, v87
	v_add_f32_e32 v147, v147, v29
	v_mul_f32_e32 v26, v30, v26
	v_mul_f32_e32 v29, v84, v93
	v_fmac_f32_e32 v25, v32, v57
	v_mul_f32_e32 v30, v84, v90
	v_mul_f32_e32 v32, v84, v88
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_add_f32_e32 v148, v148, v31
	v_fmac_f32_e32 v26, v29, v57
	v_add_f32_e32 v145, v145, v25
	v_mul_f32_e32 v25, v30, v27
	v_mul_f32_e32 v27, v32, v28
	v_mul_f32_e32 v28, v84, v91
	v_mul_f32_e32 v30, v82, v94
	v_mul_f32_e32 v31, v82, v92
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v29, v84, v89
	v_add_f32_e32 v146, v146, v26
	v_fmac_f32_e32 v25, v28, v57
	v_mul_f32_e32 v21, v30, v21
	v_mul_f32_e32 v22, v31, v22
	v_mul_f32_e32 v26, v82, v95
	v_mul_f32_e32 v28, v82, v93
	v_fmac_f32_e32 v27, v29, v57
	v_mul_f32_e32 v29, v82, v90
	v_cvt_f32_i32_e32 v23, v23
	v_fmac_f32_e32 v21, v26, v59
	v_fmac_f32_e32 v22, v28, v59
	v_add_f32_e32 v143, v143, v25
	v_mul_f32_e32 v25, v82, v91
	v_mul_f32_e32 v23, v29, v23
	v_add_f32_e32 v141, v141, v21
	v_add_f32_e32 v142, v142, v22
	v_mul_f32_e32 v21, v72, v94
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v72, v92
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v23, v25, v59
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v72, v95
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v72, v90
	v_add_f32_e32 v139, v139, v23
	v_mul_f32_e32 v23, v72, v93
	v_fmac_f32_e32 v17, v21, v56
	v_mul_f32_e32 v21, v72, v88
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v72, v91
	v_fmac_f32_e32 v18, v23, v56
	v_add_f32_e32 v137, v137, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v72, v89
	v_fmac_f32_e32 v19, v22, v56
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v86, v80
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v86, v78
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v138, v138, v18
	v_fmac_f32_e32 v17, v20, v56
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v86, v81
	v_mul_f32_e32 v19, v86, v79
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v86, v74
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v136, v136, v17
	v_fmac_f32_e32 v13, v18, v87
	v_fmac_f32_e32 v14, v19, v87
	v_mul_f32_e32 v17, v86, v75
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v84, v80
	v_mul_f32_e32 v20, v84, v78
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v133, v133, v13
	v_fmac_f32_e32 v15, v17, v87
	v_mul_f32_e32 v13, v84, v81
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v84, v79
	v_add_f32_e32 v134, v134, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v57
	v_mul_f32_e32 v13, v84, v74
	v_fmac_f32_e32 v10, v17, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v84, v76
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v84, v75
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v82, v80
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v57
	v_mul_f32_e32 v10, v82, v78
	v_mul_f32_e32 v13, v84, v77
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v82, v81
	v_add_f32_e32 v127, v127, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v82, v79
	v_mul_f32_e32 v10, v82, v74
	v_fmac_f32_e32 v5, v12, v59
	v_mul_f32_e32 v12, v82, v76
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v57
	v_fmac_f32_e32 v6, v9, v59
	v_add_f32_e32 v125, v125, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v82, v75
	v_mul_f32_e32 v9, v82, v77
	v_mul_f32_e32 v105, v116, v84
	v_mul_f32_e32 v50, v84, v98
	v_cvt_f32_i32_e32 v44, v44
	v_mul_f32_e32 v41, v72, v98
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v30, v82, v88
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v21, v86, v76
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v128, v128, v11
	v_mul_f32_e32 v10, v72, v80
	v_mul_f32_e32 v11, v72, v78
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v59
	v_fmac_f32_e32 v7, v9, v59
	v_mul_f32_e32 v8, v72, v74
	v_mul_f32_e32 v9, v72, v76
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v64, v105, v107
	v_mul_f32_e32 v85, v117, v84
	v_mul_f32_e32 v44, v50, v44
	v_mul_f32_e32 v46, v84, v99
	v_mul_f32_e32 v36, v41, v36
	v_mul_f32_e32 v38, v72, v99
	v_mul_f32_e32 v24, v30, v24
	v_mul_f32_e32 v26, v82, v89
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v86, v77
	v_add_f32_e32 v126, v126, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v72, v81
	v_mul_f32_e32 v10, v72, v79
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v72, v75
	v_mul_f32_e32 v9, v72, v77
	v_fmac_f32_e32 v64, v85, v57
	v_fmac_f32_e32 v44, v46, v57
	v_fmac_f32_e32 v36, v38, v56
	v_fmac_f32_e32 v24, v26, v59
	v_fmac_f32_e32 v16, v18, v87
	v_fmac_f32_e32 v1, v6, v56
	v_fmac_f32_e32 v2, v10, v56
	v_fmac_f32_e32 v3, v8, v56
	v_fmac_f32_e32 v4, v9, v56
	v_add_f32_e32 v177, v177, v64
	v_add_f32_e32 v172, v172, v55
	v_add_f32_e32 v164, v164, v47
	v_add_f32_e32 v160, v160, v44
	v_add_f32_e32 v151, v151, v35
	v_add_f32_e32 v152, v152, v36
	v_add_f32_e32 v144, v144, v27
	v_add_f32_e32 v140, v140, v24
	v_add_f32_e32 v131, v131, v15
	v_add_f32_e32 v132, v132, v16
	v_add_f32_e32 v124, v124, v5
	v_add_f32_e32 v123, v123, v7
	v_add_f32_e32 v120, v120, v1
	v_add_f32_e32 v122, v122, v2
	v_add_f32_e32 v67, v67, v3
	v_add_f32_e32 v121, v121, v4
	s_xor_b64 s[34:35], s[30:31], -1
	s_mov_b32 s38, 1
	s_mov_b64 s[30:31], 0
	s_and_b64 vcc, exec, s[34:35]
	s_mov_b64 s[34:35], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_10
.LBB2_13:                               ;   Parent Loop BB2_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s22, s38, s25
	s_lshl_b32 s40, s38, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[22:23], s[10:11]
	s_mov_b32 s41, s23
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[42:43], 0x48
	s_add_nc_u64 s[40:41], s[28:29], s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[42:43], s[14:15], s[42:43]
	v_add_nc_u32_e32 v78, s36, v188
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[44:45], s42, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s43, 0, s[44:45]
	v_add_co_u32 v3, s[44:45], s40, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s41, 0, s[44:45]
	v_add_co_u32 v5, s[44:45], s42, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s43, 0, s[44:45]
	v_add_co_u32 v7, s[42:43], s40, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s41, 0, s[42:43]
	global_load_b64 v[110:111], v[1:2], off offset:40
	global_load_b64 v[108:109], v[3:4], off offset:40
	global_load_b64 v[106:107], v[5:6], off offset:40
	global_load_b64 v[104:105], v[7:8], off offset:40
	v_add_nc_u32_e32 v76, s37, v187
	ds_load_2addr_stride64_b32 v[1:2], v78 offset1:2
	ds_load_2addr_stride64_b32 v[3:4], v76 offset1:2
	ds_load_2addr_stride64_b32 v[72:73], v76 offset0:4 offset1:6
	ds_load_2addr_stride64_b32 v[74:75], v78 offset0:4 offset1:6
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v1, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v1, v4, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v1, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v1, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v2, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v2, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v2, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v2, v73, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v74, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v74, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v74, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v74, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v75, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v75, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v75, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v75, v73, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v78 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[74:75], v76 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[76:77], v76 offset0:5 offset1:7
	ds_load_2addr_stride64_b32 v[78:79], v78 offset0:5 offset1:7
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v74, 0x2000, v182
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v110, s[0:1]
	v_add_nc_u32_e32 v73, 0x4000, v182
	v_cndmask_b32_e64 v75, 0, v111, s[0:1]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v74, v108, v109 offset1:16
	ds_store_2addr_b32 v73, v72, v75 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v106, s[2:3]
	v_cndmask_b32_e64 v73, 0, v107, s[2:3]
	v_add_nc_u32_e32 v74, 0x4000, v185
	v_add_nc_u32_e32 v75, 0x2000, v185
	ds_store_2addr_b32 v74, v72, v73 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v75, v104, v105 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[36:37], s[34:35], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_15
; %bb.14:                               ;   in Loop: Header=BB2_13 Depth=2
	s_add_co_i32 s22, s22, 1
	s_xor_b32 s46, s38, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[22:23], s[10:11]
	s_add_co_i32 s22, s38, s24
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[40:41], 0x48
	s_mul_u64 s[42:43], s[22:23], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[40:41], s[14:15], s[40:41]
	s_add_nc_u64 s[38:39], s[12:13], s[42:43]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v186, s[40:41]
	v_add_co_u32 v72, s[44:45], s40, v68
	s_lshl_b32 s42, s46, 6
	s_mov_b32 s43, s23
	s_mulk_i32 s22, 0x88
	v_add_co_ci_u32_e64 v73, null, s41, 0, s[44:45]
	v_add_co_u32 v76, vcc, v76, v193
	v_add_co_u32 v74, s[44:45], s40, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[38:39], s[42:43]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc
	v_add_co_u32 v82, vcc, v65, s22
	v_add_co_ci_u32_e64 v75, null, s41, 0, s[44:45]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s[40:41], s38, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s39, 0, s[40:41]
	v_add_co_u32 v80, s[40:41], s38, v71
	s_lshl_b32 s22, s46, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s39, 0, s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc, v82, s22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc
	s_clause 0x1
	global_load_b64 v[110:111], v[72:73], off offset:8
	global_load_b64 v[106:107], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[108:109], v[78:79], off offset:8
	global_load_b64 v[104:105], v[80:81], off offset:8
	global_load_b32 v194, v[76:77], off
	global_load_b32 v195, v[82:83], off
.LBB2_15:                               ; %.preheader532.i.i
                                        ;   in Loop: Header=BB2_13 Depth=2
	v_add_nc_u32_e32 v80, 0, v188
	v_add_nc_u32_e32 v81, 0, v187
	s_xor_b64 s[36:37], s[36:37], -1
	s_and_b64 s[38:39], s[30:31], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:68 offset1:70
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:36 offset1:38
	s_cselect_b32 s38, s21, 0x3c00
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:69 offset1:71
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:37 offset1:39
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s38, v191
	v_add_nc_u32_e32 v72, s22, v192
	s_and_not1_b64 vcc, exec, s[36:37]
	s_movk_i32 s36, 0x2000
	s_movk_i32 s37, 0x4000
	ds_load_2addr_b32 v[116:117], v76 offset1:1
	ds_load_2addr_b32 v[118:119], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[112:113], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[86:87], v72 offset1:1
	ds_load_2addr_b32 v[84:85], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[82:83], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[100:101], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[102:103], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[96:97], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[98:99], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[94:95], v76 offset0:64 offset1:65
	ds_load_2addr_b32 v[92:93], v76 offset0:66 offset1:67
	ds_load_2addr_b32 v[90:91], v76 offset0:68 offset1:69
	ds_load_2addr_b32 v[88:89], v76 offset0:70 offset1:71
	ds_load_2addr_b32 v[80:81], v76 offset0:96 offset1:97
	ds_load_2addr_b32 v[78:79], v76 offset0:98 offset1:99
	ds_load_2addr_b32 v[74:75], v76 offset0:100 offset1:101
	ds_load_2addr_b32 v[76:77], v76 offset0:102 offset1:103
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.16:                               ; %.preheader533.i.i
                                        ;   in Loop: Header=BB2_13 Depth=2
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v110, 0, v110, s[0:1]
	v_cndmask_b32_e64 v111, 0, v111, s[0:1]
	v_add_nc_u32_e32 v196, 0x1000, v182
	s_and_b64 s[34:35], s[34:35], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_store_2addr_b32 v182, v110, v111 offset1:16
	s_wait_loadcnt 0x3
	ds_store_2addr_b32 v196, v108, v109 offset1:16
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v108, v190, v195
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v111, s22, v189
	s_cselect_b32 s22, s21, 0x3c00
	v_cndmask_b32_e64 v106, 0, v106, s[2:3]
	v_cndmask_b32_e64 v107, 0, v107, s[2:3]
	v_cvt_f32_f16_e32 v108, v108.l
	v_cndmask_b32_e64 v110, 0, v194, s[4:5]
	v_add_nc_u32_e32 v109, 0x1000, v185
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v196, s22, v189
	s_mov_b32 s37, 0
	v_cndmask_b32_e64 v108, 0, v108, s[6:7]
	s_movk_i32 s36, 0x1000
	ds_store_2addr_b32 v185, v106, v107 offset1:16
	ds_store_2addr_b32 v109, v104, v105 offset1:16
	ds_store_b32 v111, v110
	ds_store_b32 v196, v108
	s_branch .LBB2_12
.LBB2_17:                               ; %Flow1133
	v_mov_b32_e32 v5, v179
.LBB2_18:                               ; %._crit_edge604.i.i
	v_lshrrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v3, 15, v0
	v_lshrrev_b32_e32 v6, 4, v0
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s9, s8, 31
	v_mul_u32_u24_e32 v1, 0x500, v1
	v_lshlrev_b32_e32 v2, 2, v3
	v_mul_u32_u24_e32 v0, 0x50, v3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[8:9], s[18:19]
	s_ashr_i32 s21, s20, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v8, 0, v1, v2
	s_lshl_b64 s[2:3], s[20:21], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	v_mul_lo_u32 v7, s8, v6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mad_i32_i24 v1, 0x50, v5, v8
	s_add_co_i32 s0, s20, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s0, s8
	ds_store_2addr_b32 v1, v183, v184 offset1:20
	ds_store_2addr_b32 v1, v180, v181 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshlrev_b32_e32 v1, 2, v6
	s_cselect_b64 s[0:1], -1, 0
	s_add_co_i32 s2, s18, 0x80
	v_or_b32_e32 v6, s18, v6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s2, s10
	v_add3_u32 v2, 0, v0, v1
	v_or_b32_e32 v0, s20, v3
	s_cselect_b64 s[2:3], -1, 0
	s_mov_b32 s7, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[12:13], s[0:1], s[2:3]
	v_cmp_gt_i32_e64 s[2:3], s10, v6
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_mov_b32 s6, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b64 s[14:15], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_22
; %bb.19:
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_21
; %bb.20:
	v_mad_co_i64_i32 v[9:10], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v1, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc
	global_load_b32 v1, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v4, v1
	global_store_b32 v[9:10], v1, off
.LBB2_21:                               ; %Flow1129
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[14:15], 0
.LBB2_22:                               ; %Flow1130
	v_add_lshl_u32 v3, v7, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_24
; %bb.23:
	buffer_load_b32 v1, v3, s[4:7], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v4, v1
	buffer_store_b32 v1, v3, s[4:7], null offen
.LBB2_24:
	ds_load_b32 v9, v2 offset:1280
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, 0, 1, s[12:13]
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v7, 64, v6
	s_and_not1_b64 vcc, exec, s[12:13]
	s_mov_b64 s[2:3], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_28
; %bb.25:
	v_cmp_gt_i32_e32 vcc, s10, v7
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	global_load_b32 v1, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[10:11], v1, off
.LBB2_27:                               ; %Flow1127
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_28:                               ; %Flow1128
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_30
; %bb.29:
	s_lshl_b32 s0, s8, 8
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_30:
	s_wait_dscnt 0x0
	ds_load_b32 v9, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 64, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_34
; %bb.31:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_33
; %bb.32:
	v_mad_co_i64_i32 v[10:11], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	global_load_b32 v1, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[10:11], v1, off offset:256
.LBB2_33:                               ; %Flow1125
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_34:                               ; %Flow1126
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_36
; %bb.35:
	s_movk_i32 s0, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_36:
	s_wait_dscnt 0x0
	ds_load_b32 v9, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_40
; %bb.37:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_39
; %bb.38:
	v_mad_co_i64_i32 v[10:11], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	global_load_b32 v1, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[10:11], v1, off offset:256
.LBB2_39:                               ; %Flow1123
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_40:                               ; %Flow1124
	v_mul_i32_i24_e32 v1, 0x50, v5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_42
; %bb.41:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v5, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v9, v5
	buffer_store_b32 v5, v3, s[4:7], s0 offen
.LBB2_42:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v8, v1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v8, 16, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v177, v178 offset1:20
	ds_store_2addr_b32 v5, v175, v176 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v9, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_46
; %bb.43:
	v_cmp_gt_i32_e32 vcc, s10, v8
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_45
; %bb.44:
	v_mad_co_i64_i32 v[10:11], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	global_load_b32 v1, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[10:11], v1, off
.LBB2_45:                               ; %Flow1121
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_46:                               ; %Flow1122
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_48
; %bb.47:
	s_lshl_b32 s0, s8, 6
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_48:
	ds_load_b32 v10, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 0x50, v6
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_52
; %bb.49:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v9
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_51
; %bb.50:
	v_mad_co_i64_i32 v[11:12], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	global_load_b32 v1, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	global_store_b32 v[11:12], v1, off
.LBB2_51:                               ; %Flow1119
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_52:                               ; %Flow1120
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s9, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_54
; %bb.53:
	buffer_load_b32 v1, v3, s[4:7], s9 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v3, s[4:7], s9 offen
.LBB2_54:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_58
; %bb.55:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_57
; %bb.56:
	v_mad_co_i64_i32 v[11:12], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	global_load_b32 v1, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	global_store_b32 v[11:12], v1, off offset:256
.LBB2_57:                               ; %Flow1117
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_58:                               ; %Flow1118
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_60
; %bb.59:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_60:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_64
; %bb.61:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_63
; %bb.62:
	v_mad_co_i64_i32 v[11:12], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	global_load_b32 v1, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	global_store_b32 v[11:12], v1, off offset:256
.LBB2_63:                               ; %Flow1115
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_64:                               ; %Flow1116
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_66
; %bb.65:
	s_add_co_i32 s0, s9, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_66:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v10, 32, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v173, v174 offset1:20
	ds_store_2addr_b32 v5, v171, v172 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v11, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_70
; %bb.67:
	v_cmp_gt_i32_e32 vcc, s10, v10
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_69
; %bb.68:
	v_mad_co_i64_i32 v[12:13], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v16, vcc
	global_load_b32 v1, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[12:13], v1, off
.LBB2_69:                               ; %Flow1113
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_70:                               ; %Flow1114
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_72
; %bb.71:
	s_lshl_b32 s0, s8, 7
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_72:
	ds_load_b32 v12, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 0x60, v6
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_76
; %bb.73:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v11
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_75
; %bb.74:
	v_mad_co_i64_i32 v[15:16], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	global_load_b32 v1, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[15:16], v1, off
.LBB2_75:                               ; %Flow1111
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_76:                               ; %Flow1112
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s11, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_78
; %bb.77:
	buffer_load_b32 v1, v3, s[4:7], s11 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	buffer_store_b32 v1, v3, s[4:7], s11 offen
.LBB2_78:
	s_wait_dscnt 0x0
	ds_load_b32 v12, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_82
; %bb.79:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_81
; %bb.80:
	v_mad_co_i64_i32 v[15:16], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	global_load_b32 v1, v[15:16], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[15:16], v1, off offset:256
.LBB2_81:                               ; %Flow1109
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_82:                               ; %Flow1110
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_84
; %bb.83:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_84:
	s_wait_dscnt 0x0
	ds_load_b32 v12, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_88
; %bb.85:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_87
; %bb.86:
	v_mad_co_i64_i32 v[15:16], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	global_load_b32 v1, v[15:16], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[15:16], v1, off offset:256
.LBB2_87:                               ; %Flow1107
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_88:                               ; %Flow1108
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_90
; %bb.89:
	s_add_co_i32 s0, s11, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_90:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v12, 48, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v169, v170 offset1:20
	ds_store_2addr_b32 v5, v167, v168 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_94
; %bb.91:
	v_cmp_gt_i32_e32 vcc, s10, v12
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_93
; %bb.92:
	v_mad_co_i64_i32 v[15:16], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	global_load_b32 v1, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[15:16], v1, off
.LBB2_93:                               ; %Flow1105
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_94:                               ; %Flow1106
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s12, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_96
; %bb.95:
	buffer_load_b32 v1, v3, s[4:7], s12 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	buffer_store_b32 v1, v3, s[4:7], s12 offen
.LBB2_96:
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v13, 0x70, v6
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_100
; %bb.97:
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_99
; %bb.98:
	v_mad_co_i64_i32 v[16:17], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off
.LBB2_99:                               ; %Flow1103
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_100:                              ; %Flow1104
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_mul_i32 s13, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_102
; %bb.101:
	buffer_load_b32 v1, v3, s[4:7], s13 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s13 offen
.LBB2_102:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_106
; %bb.103:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_105
; %bb.104:
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:256
.LBB2_105:                              ; %Flow1101
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_106:                              ; %Flow1102
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_108
; %bb.107:
	s_add_co_i32 s0, s12, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_108:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_112
; %bb.109:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_111
; %bb.110:
	v_mad_co_i64_i32 v[16:17], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v14, v19, vcc
	global_load_b32 v1, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:256
.LBB2_111:                              ; %Flow1099
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_112:                              ; %Flow1100
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_114
; %bb.113:
	s_add_co_i32 s0, s13, 0x100
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_114:                              ; %.preheader529.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 16, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v165, v166 offset1:20
	ds_store_2addr_b32 v5, v163, v164 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_118
; %bb.115:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_117
; %bb.116:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:64
.LBB2_117:                              ; %Flow1097
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_118:                              ; %Flow1098
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_120
; %bb.119:
	s_mov_b32 s0, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_120:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_124
; %bb.121:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_123
; %bb.122:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:64
.LBB2_123:                              ; %Flow1095
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_124:                              ; %Flow1096
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_126
; %bb.125:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_126:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x50, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_130
; %bb.127:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_129
; %bb.128:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_129:                              ; %Flow1093
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_130:                              ; %Flow1094
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_132
; %bb.131:
	s_movk_i32 s0, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_132:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_136
; %bb.133:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_135
; %bb.134:
	v_mad_co_i64_i32 v[17:18], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_135:                              ; %Flow1091
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_136:                              ; %Flow1092
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_138
; %bb.137:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_138:                              ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v161, v162 offset1:20
	ds_store_2addr_b32 v5, v159, v160 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_142
; %bb.139:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_141
; %bb.140:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_141:                              ; %Flow1089
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_142:                              ; %Flow1090
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_144
; %bb.143:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_144:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_148
; %bb.145:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_147
; %bb.146:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_147:                              ; %Flow1087
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_148:                              ; %Flow1088
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_150
; %bb.149:
	s_add_co_i32 s0, s9, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_150:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_154
; %bb.151:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_153
; %bb.152:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_153:                              ; %Flow1085
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_154:                              ; %Flow1086
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_156
; %bb.155:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_156:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_160
; %bb.157:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_159
; %bb.158:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_159:                              ; %Flow1083
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_160:                              ; %Flow1084
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_162
; %bb.161:
	s_add_co_i32 s0, s9, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_162:                              ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v157, v158 offset1:20
	ds_store_2addr_b32 v5, v155, v156 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_166
; %bb.163:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_165
; %bb.164:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_165:                              ; %Flow1081
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_166:                              ; %Flow1082
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_168
; %bb.167:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_168:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_172
; %bb.169:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_171
; %bb.170:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_171:                              ; %Flow1079
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_172:                              ; %Flow1080
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_174
; %bb.173:
	s_or_b32 s0, s11, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_174:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_178
; %bb.175:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_177
; %bb.176:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_177:                              ; %Flow1077
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_178:                              ; %Flow1078
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_180
; %bb.179:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_180:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_184
; %bb.181:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_183
; %bb.182:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:320
.LBB2_183:                              ; %Flow1075
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_184:                              ; %Flow1076
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_186
; %bb.185:
	s_add_co_i32 s0, s11, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_186:                              ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v153, v154 offset1:20
	ds_store_2addr_b32 v5, v151, v152 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_190
; %bb.187:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_189
; %bb.188:
	v_mad_co_i64_i32 v[17:18], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_189:                              ; %Flow1073
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_190:                              ; %Flow1074
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_192
; %bb.191:
	s_add_co_i32 s0, s12, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_192:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_196
; %bb.193:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_195
; %bb.194:
	v_mad_co_i64_i32 v[17:18], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v14, v20, vcc
	global_load_b32 v1, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:64
.LBB2_195:                              ; %Flow1071
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_196:                              ; %Flow1072
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_198
; %bb.197:
	s_add_co_i32 s0, s13, 64
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_198:
	ds_load_b32 v14, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_202
; %bb.199:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_201
; %bb.200:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[16:17], v1, off offset:320
.LBB2_201:                              ; %Flow1069
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_202:                              ; %Flow1070
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_204
; %bb.203:
	s_add_co_i32 s0, s12, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_204:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_208
; %bb.205:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_207
; %bb.206:
	v_mad_co_i64_i32 v[15:16], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	global_load_b32 v1, v[15:16], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[15:16], v1, off offset:320
.LBB2_207:                              ; %Flow1067
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_208:                              ; %Flow1068
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_210
; %bb.209:
	s_add_co_i32 s0, s13, 0x140
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_210:                              ; %.preheader529.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 32, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v149, v150 offset1:20
	ds_store_2addr_b32 v5, v147, v148 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_214
; %bb.211:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_213
; %bb.212:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:128
.LBB2_213:                              ; %Flow1065
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_214:                              ; %Flow1066
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_216
; %bb.215:
	s_movk_i32 s0, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_216:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_220
; %bb.217:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_219
; %bb.218:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:128
.LBB2_219:                              ; %Flow1063
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_220:                              ; %Flow1064
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_222
; %bb.221:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_bitset1_b32 s0, 7
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_222:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x60, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB2_226
; %bb.223:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_225
; %bb.224:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_225:                              ; %Flow1061
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_226:                              ; %Flow1062
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_228
; %bb.227:
	s_movk_i32 s0, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_228:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_232
; %bb.229:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_231
; %bb.230:
	v_mad_co_i64_i32 v[17:18], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_231:                              ; %Flow1059
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_232:                              ; %Flow1060
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_234
; %bb.233:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_234:                              ; %.preheader.1.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v145, v146 offset1:20
	ds_store_2addr_b32 v5, v143, v144 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_238
; %bb.235:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_237
; %bb.236:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_237:                              ; %Flow1057
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_238:                              ; %Flow1058
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_240
; %bb.239:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_240:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_244
; %bb.241:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_243
; %bb.242:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_243:                              ; %Flow1055
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_244:                              ; %Flow1056
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_246
; %bb.245:
	s_add_co_i32 s0, s9, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_246:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_250
; %bb.247:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_249
; %bb.248:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_249:                              ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_250:                              ; %Flow1054
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_252
; %bb.251:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_252:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_256
; %bb.253:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_255
; %bb.254:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_255:                              ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_256:                              ; %Flow1052
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_258
; %bb.257:
	s_add_co_i32 s0, s9, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_258:                              ; %.preheader.2.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v141, v142 offset1:20
	ds_store_2addr_b32 v5, v139, v140 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_262
; %bb.259:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_261
; %bb.260:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_261:                              ; %Flow1049
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_262:                              ; %Flow1050
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_264
; %bb.263:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_264:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_268
; %bb.265:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_267
; %bb.266:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_267:                              ; %Flow1047
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_268:                              ; %Flow1048
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_270
; %bb.269:
	s_add_co_i32 s0, s11, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_270:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_274
; %bb.271:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_273
; %bb.272:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_273:                              ; %Flow1045
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_274:                              ; %Flow1046
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_276
; %bb.275:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_276:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_280
; %bb.277:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_279
; %bb.278:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:384
.LBB2_279:                              ; %Flow1043
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_280:                              ; %Flow1044
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_282
; %bb.281:
	s_add_co_i32 s0, s11, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_282:                              ; %.preheader.3.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v137, v138 offset1:20
	ds_store_2addr_b32 v5, v135, v136 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_286
; %bb.283:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_285
; %bb.284:
	v_mad_co_i64_i32 v[17:18], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_285:                              ; %Flow1041
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_286:                              ; %Flow1042
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_288
; %bb.287:
	s_add_co_i32 s0, s12, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_288:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_292
; %bb.289:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_291
; %bb.290:
	v_mad_co_i64_i32 v[17:18], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v14, v20, vcc
	global_load_b32 v1, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:128
.LBB2_291:                              ; %Flow1039
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_292:                              ; %Flow1040
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_294
; %bb.293:
	s_add_co_i32 s0, s13, 0x80
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_294:
	ds_load_b32 v14, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_298
; %bb.295:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_297
; %bb.296:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[16:17], v1, off offset:384
.LBB2_297:                              ; %Flow1037
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_298:                              ; %Flow1038
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_300
; %bb.299:
	s_add_co_i32 s0, s12, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_300:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_304
; %bb.301:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_303
; %bb.302:
	v_mad_co_i64_i32 v[15:16], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	global_load_b32 v1, v[15:16], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[15:16], v1, off offset:384
.LBB2_303:                              ; %Flow1035
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_304:                              ; %Flow1036
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_306
; %bb.305:
	s_add_co_i32 s0, s13, 0x180
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_306:                              ; %.preheader529.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 48, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v133, v134 offset1:20
	ds_store_2addr_b32 v5, v131, v132 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_310
; %bb.307:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_309
; %bb.308:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:192
.LBB2_309:                              ; %Flow1033
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB2_310:                              ; %Flow1034
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_312
; %bb.311:
	s_movk_i32 s0, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_312:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_316
; %bb.313:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_315
; %bb.314:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	global_load_b32 v1, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	global_store_b32 v[16:17], v1, off offset:192
.LBB2_315:                              ; %Flow1031
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_316:                              ; %Flow1032
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_318
; %bb.317:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v15, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_318:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x70, v0
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_322
; %bb.319:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_321
; %bb.320:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v6, v20, vcc
	global_load_b32 v1, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	global_store_b32 v[17:18], v1, off offset:448
.LBB2_321:                              ; %Flow1029
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_322:                              ; %Flow1030
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_324
; %bb.323:
	s_movk_i32 s0, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v16, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_324:
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_328
; %bb.325:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_327
; %bb.326:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	global_load_b32 v1, v[16:17], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[16:17], v1, off offset:448
.LBB2_327:                              ; %Flow1027
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_328:                              ; %Flow1028
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_330
; %bb.329:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_330:                              ; %.preheader.1.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v129, v130 offset1:20
	ds_store_2addr_b32 v5, v127, v128 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_334
; %bb.331:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_333
; %bb.332:
	v_mad_co_i64_i32 v[16:17], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	global_load_b32 v1, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[16:17], v1, off offset:192
.LBB2_333:                              ; %Flow1025
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_334:                              ; %Flow1026
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_336
; %bb.335:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_336:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_340
; %bb.337:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_339
; %bb.338:
	v_mad_co_i64_i32 v[16:17], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	global_load_b32 v1, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[16:17], v1, off offset:192
.LBB2_339:                              ; %Flow1023
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_340:                              ; %Flow1024
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_342
; %bb.341:
	s_add_co_i32 s0, s9, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_342:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_346
; %bb.343:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_345
; %bb.344:
	v_mad_co_i64_i32 v[7:8], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	global_load_b32 v1, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:448
.LBB2_345:                              ; %Flow1021
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_346:                              ; %Flow1022
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_348
; %bb.347:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_348:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_352
; %bb.349:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_351
; %bb.350:
	v_mad_co_i64_i32 v[7:8], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	global_load_b32 v1, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:448
.LBB2_351:                              ; %Flow1019
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_352:                              ; %Flow1020
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_354
; %bb.353:
	s_addk_co_i32 s9, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s9 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s9 offen
.LBB2_354:                              ; %.preheader.2.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v125, v126 offset1:20
	ds_store_2addr_b32 v5, v124, v123 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_358
; %bb.355:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_357
; %bb.356:
	v_mad_co_i64_i32 v[7:8], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	global_load_b32 v1, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:192
.LBB2_357:                              ; %Flow1017
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_358:                              ; %Flow1018
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_360
; %bb.359:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_360:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_364
; %bb.361:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_363
; %bb.362:
	v_mad_co_i64_i32 v[7:8], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	global_load_b32 v1, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:192
.LBB2_363:                              ; %Flow1015
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_364:                              ; %Flow1016
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_366
; %bb.365:
	s_add_co_i32 s0, s11, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_366:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_370
; %bb.367:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_369
; %bb.368:
	v_mad_co_i64_i32 v[7:8], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	global_load_b32 v1, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:448
.LBB2_369:                              ; %Flow1013
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_370:                              ; %Flow1014
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_372
; %bb.371:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_372:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_376
; %bb.373:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_375
; %bb.374:
	v_mad_co_i64_i32 v[7:8], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	global_load_b32 v1, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	global_store_b32 v[7:8], v1, off offset:448
.LBB2_375:                              ; %Flow1011
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_376:                              ; %Flow1012
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_378
; %bb.377:
	s_addk_co_i32 s11, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s11 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v6, v1
	buffer_store_b32 v1, v3, s[4:7], s11 offen
.LBB2_378:                              ; %.preheader.3.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v120, v122 offset1:20
	ds_store_2addr_b32 v5, v67, v121 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v5, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_382
; %bb.379:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_381
; %bb.380:
	v_mad_co_i64_i32 v[6:7], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	global_load_b32 v1, v[6:7], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	global_store_b32 v[6:7], v1, off offset:192
.LBB2_381:                              ; %Flow1009
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_382:                              ; %Flow1010
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_384
; %bb.383:
	s_add_co_i32 s0, s12, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_384:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_388
; %bb.385:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_387
; %bb.386:
	v_mad_co_i64_i32 v[6:7], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	global_load_b32 v1, v[6:7], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	global_store_b32 v[6:7], v1, off offset:192
.LBB2_387:                              ; %Flow1007
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_388:                              ; %Flow1008
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_390
; %bb.389:
	s_add_co_i32 s0, s13, 0xc0
	buffer_load_b32 v1, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	buffer_store_b32 v1, v3, s[4:7], s0 offen
.LBB2_390:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_394
; %bb.391:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_393
; %bb.392:
	v_mad_co_i64_i32 v[6:7], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	global_load_b32 v1, v[6:7], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	global_store_b32 v[6:7], v1, off offset:448
.LBB2_393:                              ; %Flow1005
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_394:                              ; %Flow1006
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_396
; %bb.395:
	s_addk_co_i32 s12, 0x1c0
	buffer_load_b32 v1, v3, s[4:7], s12 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v5, v1
	buffer_store_b32 v1, v3, s[4:7], s12 offen
.LBB2_396:
	ds_load_b32 v2, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB2_400
; %bb.397:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB2_399
; %bb.398:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[4:5], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc
	global_load_b32 v4, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v2, v4
	global_store_b32 v[0:1], v4, off offset:448
.LBB2_399:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB2_400:                              ; %Flow1004
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_402
; %bb.401:
	s_addk_co_i32 s13, 0x1c0
	buffer_load_b32 v0, v3, s[4:7], s13 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v2, v0
	buffer_store_b32 v0, v3, s[4:7], s13 offen
.LBB2_402:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_403:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.Lfunc_end2:
	.size	iu4_w64_r4_add, .Lfunc_end2-iu4_w64_r4_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r4_add
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
		.amdhsa_next_free_vgpr 197
		.amdhsa_next_free_sgpr 47
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-iu4_w64_r4_add)<<4)&4080)>>4
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
	.set .Liu4_w64_r4_add.num_vgpr, 197
	.set .Liu4_w64_r4_add.num_agpr, 0
	.set .Liu4_w64_r4_add.numbered_sgpr, 47
	.set .Liu4_w64_r4_add.num_named_barrier, 0
	.set .Liu4_w64_r4_add.private_seg_size, 0
	.set .Liu4_w64_r4_add.uses_vcc, 1
	.set .Liu4_w64_r4_add.uses_flat_scratch, 0
	.set .Liu4_w64_r4_add.has_dyn_sized_stack, 0
	.set .Liu4_w64_r4_add.has_recursion, 0
	.set .Liu4_w64_r4_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20896
; TotalNumSgprs: 49
; NumVgprs: 197
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 49
; NumSGPRsForWavesPerEU: 49
; NumVGPRsForWavesPerEU: 197
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	iu4_w64_r4_set          ; -- Begin function iu4_w64_r4_set
	.globl	iu4_w64_r4_set
	.p2align	8
	.type	iu4_w64_r4_set,@function
iu4_w64_r4_set:                         ; @iu4_w64_r4_set
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[8:10], s[0:1], 0x18
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s20, s8
	s_cselect_b64 s[2:3], -1, 0
	s_cmp_ge_i32 s18, s10
	s_cselect_b64 s[4:5], -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b64 s[2:3], s[2:3], s[4:5]
	s_and_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB3_403
; %bb.1:                                ; %.preheader533.i.i
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s9, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s9, s0
	s_add_co_i32 s7, s8, -1
	s_ashr_i32 s19, s0, 8
	s_mov_b64 s[0:1], exec
	s_mul_i32 s6, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i.i
	v_or_b32_e32 v5, s18, v2
	v_mov_b32_e32 v4, 0
	v_lshlrev_b32_e32 v3, 2, v1
	s_mov_b64 s[2:3], exec
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s10, v5
	s_cbranch_execz .LBB3_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc
	global_load_b32 v4, v[4:5], off
.LBB3_4:                                ; %.preheader530.loopexit.i.i
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v7, s20, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s7, v7
	v_cmp_gt_i32_e32 vcc, s8, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s6, v5, s[12:13]
	global_load_b32 v5, v[5:6], off
	v_lshlrev_b32_e32 v6, 4, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v5, v6, v5
	v_lshlrev_b32_e32 v6, 3, v2
	v_cvt_f32_f16_e32 v5, v5.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v3, 0, v6, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, 0, v5, vcc
	ds_store_2addr_stride64_b32 v3, v4, v5 offset0:48 offset1:56
.LBB3_5:                                ; %Flow1137
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v3, 2, v0
	s_add_co_i32 s21, s10, -1
	v_lshrrev_b32_e32 v13, 5, v0
	v_lshlrev_b32_e32 v14, 7, v0
	v_and_b32_e32 v15, 60, v0
	v_add_nc_u32_e32 v11, s18, v3
	v_add_nc_u32_e32 v4, s20, v3
	v_and_b32_e32 v3, 3, v0
	v_bfe_u32 v16, v0, 1, 1
	v_or_b32_e32 v17, 8, v13
	v_add_nc_u32_e32 v12, 64, v11
	v_add_nc_u32_e32 v5, 64, v4
	v_lshlrev_b32_e32 v3, 3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v6, s21, v11
	v_min_i32_e32 v4, s7, v4
	v_min_i32_e32 v7, s21, v12
	v_min_i32_e32 v5, s7, v5
	v_and_or_b32 v14, 0x80, v14, v15
	v_and_or_b32 v13, v13, 6, v16
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s6, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s6, v5, v[3:4]
	v_and_or_b32 v15, v17, 14, v16
	v_lshl_or_b32 v13, v13, 8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[14:15] offset:8
	global_load_b64 v[5:6], v69, s[12:13] offset:8
	global_load_b64 v[7:8], v70, s[14:15] offset:8
	global_load_b64 v[9:10], v71, s[12:13] offset:8
	v_lshl_or_b32 v14, v15, 8, v14
	v_add_nc_u32_e32 v182, 0, v13
	v_cmp_gt_i32_e64 s[2:3], s10, v12
	s_cmp_gt_i32 s9, 0xff
	v_add_nc_u32_e32 v185, 0, v14
	v_add_nc_u32_e32 v11, 0x1000, v182
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v12, 0x1000, v185
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v3, 0, v3, s[0:1]
	v_cndmask_b32_e64 v4, 0, v4, s[0:1]
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v7, 0, v7, s[2:3]
	v_cndmask_b32_e64 v8, 0, v8, s[2:3]
	ds_store_2addr_b32 v11, v5, v6 offset1:16
	ds_store_2addr_b32 v182, v3, v4 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v12, v9, v10 offset1:16
	ds_store_2addr_b32 v185, v7, v8 offset1:16
	v_bfe_u32 v3, v0, 4, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v179, 2, v3
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_7
; %bb.6:                                ; %._crit_edge.._crit_edge599_crit_edge.i.i
	v_lshlrev_b32_e32 v5, 2, v3
	s_mov_b64 s[4:5], 0
	s_branch .LBB3_8
.LBB3_7:
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr5
.LBB3_8:                                ; %Flow1135
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_and_not1_b64 vcc, exec, s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_18
; %bb.9:                                ; %.preheader529.lr.ph.i.i
	v_add_nc_u32_e32 v3, s20, v2
	v_and_b32_e32 v4, 63, v0
	v_and_b32_e32 v5, 64, v2
	v_add_nc_u32_e32 v8, s18, v2
	v_lshlrev_b32_e32 v2, 3, v2
	v_min_i32_e32 v7, s7, v3
	v_lshlrev_b32_e32 v9, 2, v1
	v_lshlrev_b32_e32 v10, 3, v0
	v_lshlrev_b32_e32 v6, 5, v0
	v_lshlrev_b32_e32 v4, 2, v4
	v_mad_co_u64_u32 v[65:66], null, s6, v7, s[12:13]
	v_ashrrev_i32_e32 v7, 31, v7
	v_or_b32_e32 v11, v179, v5
	v_add3_u32 v189, 0, v2, v9
	v_and_b32_e32 v2, 0x278, v10
	v_mov_b32_e32 v120, 0
	v_min_i32_e32 v186, s21, v8
	v_and_or_b32 v187, 0x800, v6, v4
	v_mad_co_u64_u32 v[66:67], null, s6, v7, v[66:67]
	v_lshl_or_b32 v188, v5, 5, v4
	v_cmp_gt_i32_e64 s[4:5], s10, v8
	v_cmp_gt_i32_e64 s[6:7], s8, v3
	v_lshlrev_b32_e32 v190, 4, v1
	v_lshl_add_u32 v191, v11, 3, 0
	v_add_nc_u32_e32 v192, 0, v2
	v_lshlrev_b32_e32 v193, 2, v1
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	s_mov_b32 s23, 0
	s_ashr_i32 s11, s10, 31
	s_movk_i32 s36, 0x1000
	s_movk_i32 s9, 0x3000
	s_movk_i32 s21, 0x3800
	s_mov_b32 s37, 0
	s_mov_b32 s24, s23
	s_branch .LBB3_11
.LBB3_10:                               ;   in Loop: Header=BB3_11 Depth=1
	s_and_b64 vcc, exec, s[26:27]
	s_mov_b32 s24, s33
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_17
.LBB3_11:                               ; %.preheader529.i.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_13 Depth 2
	s_mov_b32 s25, s23
	s_add_co_i32 s33, s24, 1
	s_mul_u64 s[28:29], s[24:25], 0x88
	s_lshl_b32 s25, s24, 1
	s_cmp_eq_u32 s33, s19
	s_mov_b64 s[34:35], 0
	s_cselect_b64 s[26:27], -1, 0
	s_add_nc_u64 s[28:29], s[12:13], s[28:29]
	s_mov_b64 s[30:31], -1
	s_mov_b32 s38, s23
	s_branch .LBB3_13
.LBB3_12:                               ;   in Loop: Header=BB3_13 Depth=2
	s_wait_loadcnt_dscnt 0x20f
	v_mul_f32_e32 v104, v116, v86
	v_cvt_f32_i32_e32 v61, v61
	v_cvt_f32_i32_e32 v87, v87
	v_mul_f32_e32 v105, v118, v86
	v_cvt_f32_i32_e32 v62, v62
	v_mul_f32_e32 v106, v117, v86
	v_mul_f32_e32 v61, v104, v61
	v_mul_f32_e32 v104, v119, v86
	v_mul_f32_e32 v107, v114, v86
	v_mul_f32_e32 v62, v105, v62
	v_mul_f32_e32 v105, v112, v86
	v_fmac_f32_e32 v61, v106, v87
	v_cvt_f32_i32_e32 v63, v63
	v_cvt_f32_i32_e32 v64, v64
	v_fmac_f32_e32 v62, v104, v87
	v_mul_f32_e32 v104, v113, v86
	v_add_f32_e32 v183, v183, v61
	v_mul_f32_e32 v61, v107, v63
	v_mul_f32_e32 v63, v105, v64
	v_mul_f32_e32 v64, v115, v86
	s_wait_dscnt 0xe
	v_mul_f32_e32 v106, v118, v84
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v107, v57
	v_fmac_f32_e32 v63, v104, v87
	v_fmac_f32_e32 v61, v64, v87
	v_cvt_f32_i32_e32 v57, v85
	v_mul_f32_e32 v58, v106, v58
	v_mul_f32_e32 v104, v119, v84
	v_add_f32_e32 v184, v184, v62
	v_add_f32_e32 v180, v180, v61
	v_mul_f32_e32 v61, v114, v84
	v_cvt_f32_i32_e32 v59, v59
	v_fmac_f32_e32 v58, v104, v57
	v_mul_f32_e32 v62, v112, v84
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v53, v53
	v_add_f32_e32 v181, v181, v63
	v_add_f32_e32 v178, v178, v58
	v_mul_f32_e32 v58, v61, v59
	v_mul_f32_e32 v59, v115, v84
	v_mul_f32_e32 v60, v62, v60
	s_wait_dscnt 0xd
	v_mul_f32_e32 v62, v116, v82
	v_mul_f32_e32 v61, v113, v84
	v_mul_f32_e32 v63, v118, v82
	v_fmac_f32_e32 v58, v59, v57
	v_cvt_f32_i32_e32 v59, v83
	v_cvt_f32_i32_e32 v54, v54
	v_mul_f32_e32 v53, v62, v53
	v_mul_f32_e32 v62, v117, v82
	v_fmac_f32_e32 v60, v61, v57
	v_add_f32_e32 v175, v175, v58
	v_mul_f32_e32 v54, v63, v54
	v_mul_f32_e32 v58, v119, v82
	v_fmac_f32_e32 v53, v62, v59
	v_mul_f32_e32 v61, v114, v82
	v_mul_f32_e32 v62, v112, v82
	v_cvt_f32_i32_e32 v55, v55
	v_cvt_f32_i32_e32 v56, v56
	v_add_f32_e32 v176, v176, v60
	v_fmac_f32_e32 v54, v58, v59
	v_add_f32_e32 v173, v173, v53
	v_mul_f32_e32 v53, v61, v55
	v_mul_f32_e32 v55, v62, v56
	v_mul_f32_e32 v56, v115, v82
	v_mul_f32_e32 v58, v113, v82
	s_wait_dscnt 0xc
	v_mul_f32_e32 v60, v116, v72
	v_mul_f32_e32 v61, v118, v72
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_fmac_f32_e32 v53, v56, v59
	v_fmac_f32_e32 v55, v58, v59
	v_cvt_f32_i32_e32 v56, v73
	v_mul_f32_e32 v49, v60, v49
	v_mul_f32_e32 v50, v61, v50
	v_mul_f32_e32 v58, v117, v72
	v_mul_f32_e32 v60, v119, v72
	v_add_f32_e32 v174, v174, v54
	v_add_f32_e32 v171, v171, v53
	v_mul_f32_e32 v53, v114, v72
	v_fmac_f32_e32 v49, v58, v56
	v_fmac_f32_e32 v50, v60, v56
	v_cvt_f32_i32_e32 v51, v51
	v_mul_f32_e32 v54, v112, v72
	v_cvt_f32_i32_e32 v52, v52
	v_add_f32_e32 v169, v169, v49
	v_add_f32_e32 v170, v170, v50
	v_mul_f32_e32 v49, v53, v51
	v_mul_f32_e32 v50, v115, v72
	v_mul_f32_e32 v51, v54, v52
	s_wait_dscnt 0xb
	v_mul_f32_e32 v52, v86, v100
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v53, v113, v72
	v_fmac_f32_e32 v49, v50, v56
	s_wait_dscnt 0xa
	v_mul_f32_e32 v50, v86, v102
	v_cvt_f32_i32_e32 v46, v46
	v_mul_f32_e32 v45, v52, v45
	v_mul_f32_e32 v52, v86, v101
	v_fmac_f32_e32 v51, v53, v56
	v_add_f32_e32 v167, v167, v49
	v_mul_f32_e32 v46, v50, v46
	v_mul_f32_e32 v49, v86, v103
	v_fmac_f32_e32 v45, v52, v87
	s_wait_dscnt 0x9
	v_mul_f32_e32 v50, v86, v96
	s_wait_dscnt 0x8
	v_mul_f32_e32 v52, v86, v98
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_add_f32_e32 v168, v168, v51
	v_fmac_f32_e32 v46, v49, v87
	v_add_f32_e32 v165, v165, v45
	v_mul_f32_e32 v45, v50, v47
	v_mul_f32_e32 v47, v52, v48
	v_mul_f32_e32 v48, v86, v97
	v_mul_f32_e32 v50, v84, v100
	v_mul_f32_e32 v51, v84, v102
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_mul_f32_e32 v49, v86, v99
	v_add_f32_e32 v166, v166, v46
	v_fmac_f32_e32 v45, v48, v87
	v_mul_f32_e32 v41, v50, v41
	v_mul_f32_e32 v42, v51, v42
	v_mul_f32_e32 v46, v84, v101
	v_mul_f32_e32 v48, v84, v103
	v_fmac_f32_e32 v47, v49, v87
	v_mul_f32_e32 v49, v84, v96
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v46, v57
	v_fmac_f32_e32 v42, v48, v57
	v_add_f32_e32 v163, v163, v45
	v_mul_f32_e32 v45, v84, v97
	v_mul_f32_e32 v43, v49, v43
	v_add_f32_e32 v161, v161, v41
	v_add_f32_e32 v162, v162, v42
	v_mul_f32_e32 v41, v82, v100
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v42, v82, v102
	v_cvt_f32_i32_e32 v38, v38
	v_fmac_f32_e32 v43, v45, v57
	v_cvt_f32_i32_e32 v39, v39
	v_mul_f32_e32 v37, v41, v37
	v_mul_f32_e32 v41, v82, v101
	v_mul_f32_e32 v38, v42, v38
	v_mul_f32_e32 v42, v82, v96
	v_add_f32_e32 v159, v159, v43
	v_mul_f32_e32 v43, v82, v103
	v_fmac_f32_e32 v37, v41, v59
	v_mul_f32_e32 v41, v82, v98
	v_cvt_f32_i32_e32 v40, v40
	v_mul_f32_e32 v39, v42, v39
	v_mul_f32_e32 v42, v82, v97
	v_fmac_f32_e32 v38, v43, v59
	v_add_f32_e32 v157, v157, v37
	v_mul_f32_e32 v37, v41, v40
	v_mul_f32_e32 v40, v82, v99
	v_fmac_f32_e32 v39, v42, v59
	v_mul_f32_e32 v41, v72, v100
	v_mul_f32_e32 v42, v72, v102
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v158, v158, v38
	v_fmac_f32_e32 v37, v40, v59
	v_add_f32_e32 v155, v155, v39
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v38, v72, v101
	v_mul_f32_e32 v39, v72, v103
	v_mul_f32_e32 v40, v72, v96
	v_cvt_f32_i32_e32 v35, v35
	v_add_f32_e32 v156, v156, v37
	v_fmac_f32_e32 v33, v38, v56
	v_fmac_f32_e32 v34, v39, v56
	v_mul_f32_e32 v37, v72, v97
	v_mul_f32_e32 v35, v40, v35
	s_wait_dscnt 0x7
	v_mul_f32_e32 v39, v86, v94
	s_wait_dscnt 0x6
	v_mul_f32_e32 v40, v86, v92
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_add_f32_e32 v153, v153, v33
	v_fmac_f32_e32 v35, v37, v56
	v_mul_f32_e32 v33, v86, v95
	v_mul_f32_e32 v29, v39, v29
	v_mul_f32_e32 v30, v40, v30
	v_mul_f32_e32 v37, v86, v93
	v_add_f32_e32 v154, v154, v34
	v_cvt_f32_i32_e32 v31, v31
	v_fmac_f32_e32 v29, v33, v87
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v86, v90
	v_fmac_f32_e32 v30, v37, v87
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v86, v88
	v_cvt_f32_i32_e32 v32, v32
	v_add_f32_e32 v149, v149, v29
	v_mul_f32_e32 v29, v33, v31
	v_add_f32_e32 v150, v150, v30
	v_mul_f32_e32 v30, v86, v91
	v_mul_f32_e32 v31, v34, v32
	v_mul_f32_e32 v32, v84, v94
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v33, v86, v89
	v_fmac_f32_e32 v29, v30, v87
	v_mul_f32_e32 v30, v84, v92
	v_cvt_f32_i32_e32 v26, v26
	v_mul_f32_e32 v25, v32, v25
	v_mul_f32_e32 v32, v84, v95
	v_fmac_f32_e32 v31, v33, v87
	v_add_f32_e32 v147, v147, v29
	v_mul_f32_e32 v26, v30, v26
	v_mul_f32_e32 v29, v84, v93
	v_fmac_f32_e32 v25, v32, v57
	v_mul_f32_e32 v30, v84, v90
	v_mul_f32_e32 v32, v84, v88
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_add_f32_e32 v148, v148, v31
	v_fmac_f32_e32 v26, v29, v57
	v_add_f32_e32 v145, v145, v25
	v_mul_f32_e32 v25, v30, v27
	v_mul_f32_e32 v27, v32, v28
	v_mul_f32_e32 v28, v84, v91
	v_mul_f32_e32 v30, v82, v94
	v_mul_f32_e32 v31, v82, v92
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_mul_f32_e32 v29, v84, v89
	v_add_f32_e32 v146, v146, v26
	v_fmac_f32_e32 v25, v28, v57
	v_mul_f32_e32 v21, v30, v21
	v_mul_f32_e32 v22, v31, v22
	v_mul_f32_e32 v26, v82, v95
	v_mul_f32_e32 v28, v82, v93
	v_fmac_f32_e32 v27, v29, v57
	v_mul_f32_e32 v29, v82, v90
	v_cvt_f32_i32_e32 v23, v23
	v_fmac_f32_e32 v21, v26, v59
	v_fmac_f32_e32 v22, v28, v59
	v_add_f32_e32 v143, v143, v25
	v_mul_f32_e32 v25, v82, v91
	v_mul_f32_e32 v23, v29, v23
	v_add_f32_e32 v141, v141, v21
	v_add_f32_e32 v142, v142, v22
	v_mul_f32_e32 v21, v72, v94
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v22, v72, v92
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v23, v25, v59
	v_cvt_f32_i32_e32 v19, v19
	v_mul_f32_e32 v17, v21, v17
	v_mul_f32_e32 v21, v72, v95
	v_mul_f32_e32 v18, v22, v18
	v_mul_f32_e32 v22, v72, v90
	v_add_f32_e32 v139, v139, v23
	v_mul_f32_e32 v23, v72, v93
	v_fmac_f32_e32 v17, v21, v56
	v_mul_f32_e32 v21, v72, v88
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v19, v22, v19
	v_mul_f32_e32 v22, v72, v91
	v_fmac_f32_e32 v18, v23, v56
	v_add_f32_e32 v137, v137, v17
	v_mul_f32_e32 v17, v21, v20
	v_mul_f32_e32 v20, v72, v89
	v_fmac_f32_e32 v19, v22, v56
	s_wait_dscnt 0x3
	v_mul_f32_e32 v21, v86, v80
	s_wait_dscnt 0x2
	v_mul_f32_e32 v22, v86, v78
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_add_f32_e32 v138, v138, v18
	v_fmac_f32_e32 v17, v20, v56
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v13, v21, v13
	v_mul_f32_e32 v14, v22, v14
	v_mul_f32_e32 v18, v86, v81
	v_mul_f32_e32 v19, v86, v79
	s_wait_dscnt 0x1
	v_mul_f32_e32 v20, v86, v74
	v_cvt_f32_i32_e32 v15, v15
	v_add_f32_e32 v136, v136, v17
	v_fmac_f32_e32 v13, v18, v87
	v_fmac_f32_e32 v14, v19, v87
	v_mul_f32_e32 v17, v86, v75
	v_mul_f32_e32 v15, v20, v15
	v_mul_f32_e32 v19, v84, v80
	v_mul_f32_e32 v20, v84, v78
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_add_f32_e32 v133, v133, v13
	v_fmac_f32_e32 v15, v17, v87
	v_mul_f32_e32 v13, v84, v81
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v10, v20, v10
	v_mul_f32_e32 v17, v84, v79
	v_add_f32_e32 v134, v134, v14
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v9, v13, v57
	v_mul_f32_e32 v13, v84, v74
	v_fmac_f32_e32 v10, v17, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v14, v84, v76
	v_cvt_f32_i32_e32 v12, v12
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v13, v11
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v84, v75
	v_mul_f32_e32 v11, v14, v12
	v_mul_f32_e32 v12, v82, v80
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v9, v10, v57
	v_mul_f32_e32 v10, v82, v78
	v_mul_f32_e32 v13, v84, v77
	v_mul_f32_e32 v5, v12, v5
	v_mul_f32_e32 v12, v82, v81
	v_add_f32_e32 v127, v127, v9
	v_mul_f32_e32 v6, v10, v6
	v_mul_f32_e32 v9, v82, v79
	v_mul_f32_e32 v10, v82, v74
	v_fmac_f32_e32 v5, v12, v59
	v_mul_f32_e32 v12, v82, v76
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v11, v13, v57
	v_fmac_f32_e32 v6, v9, v59
	v_add_f32_e32 v125, v125, v5
	v_mul_f32_e32 v5, v10, v7
	v_mul_f32_e32 v7, v12, v8
	v_mul_f32_e32 v8, v82, v75
	v_mul_f32_e32 v9, v82, v77
	v_mul_f32_e32 v105, v116, v84
	v_mul_f32_e32 v50, v84, v98
	v_cvt_f32_i32_e32 v44, v44
	v_mul_f32_e32 v41, v72, v98
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v30, v82, v88
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v21, v86, v76
	v_cvt_f32_i32_e32 v16, v16
	v_add_f32_e32 v128, v128, v11
	v_mul_f32_e32 v10, v72, v80
	v_mul_f32_e32 v11, v72, v78
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v5, v8, v59
	v_fmac_f32_e32 v7, v9, v59
	v_mul_f32_e32 v8, v72, v74
	v_mul_f32_e32 v9, v72, v76
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_mul_f32_e32 v64, v105, v107
	v_mul_f32_e32 v85, v117, v84
	v_mul_f32_e32 v44, v50, v44
	v_mul_f32_e32 v46, v84, v99
	v_mul_f32_e32 v36, v41, v36
	v_mul_f32_e32 v38, v72, v99
	v_mul_f32_e32 v24, v30, v24
	v_mul_f32_e32 v26, v82, v89
	v_mul_f32_e32 v16, v21, v16
	v_mul_f32_e32 v18, v86, v77
	v_add_f32_e32 v126, v126, v6
	v_mul_f32_e32 v1, v10, v1
	v_mul_f32_e32 v2, v11, v2
	v_mul_f32_e32 v6, v72, v81
	v_mul_f32_e32 v10, v72, v79
	v_mul_f32_e32 v3, v8, v3
	v_mul_f32_e32 v4, v9, v4
	v_mul_f32_e32 v8, v72, v75
	v_mul_f32_e32 v9, v72, v77
	v_fmac_f32_e32 v64, v85, v57
	v_fmac_f32_e32 v44, v46, v57
	v_fmac_f32_e32 v36, v38, v56
	v_fmac_f32_e32 v24, v26, v59
	v_fmac_f32_e32 v16, v18, v87
	v_fmac_f32_e32 v1, v6, v56
	v_fmac_f32_e32 v2, v10, v56
	v_fmac_f32_e32 v3, v8, v56
	v_fmac_f32_e32 v4, v9, v56
	v_add_f32_e32 v177, v177, v64
	v_add_f32_e32 v172, v172, v55
	v_add_f32_e32 v164, v164, v47
	v_add_f32_e32 v160, v160, v44
	v_add_f32_e32 v151, v151, v35
	v_add_f32_e32 v152, v152, v36
	v_add_f32_e32 v144, v144, v27
	v_add_f32_e32 v140, v140, v24
	v_add_f32_e32 v131, v131, v15
	v_add_f32_e32 v132, v132, v16
	v_add_f32_e32 v124, v124, v5
	v_add_f32_e32 v123, v123, v7
	v_add_f32_e32 v120, v120, v1
	v_add_f32_e32 v122, v122, v2
	v_add_f32_e32 v67, v67, v3
	v_add_f32_e32 v121, v121, v4
	s_xor_b64 s[34:35], s[30:31], -1
	s_mov_b32 s38, 1
	s_mov_b64 s[30:31], 0
	s_and_b64 vcc, exec, s[34:35]
	s_mov_b64 s[34:35], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_10
.LBB3_13:                               ;   Parent Loop BB3_11 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s22, s38, s25
	s_lshl_b32 s40, s38, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[22:23], s[10:11]
	s_mov_b32 s41, s23
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[42:43], s[42:43], 0x48
	s_add_nc_u64 s[40:41], s[28:29], s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[42:43], s[14:15], s[42:43]
	v_add_nc_u32_e32 v78, s36, v188
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s[44:45], s42, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s43, 0, s[44:45]
	v_add_co_u32 v3, s[44:45], s40, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s41, 0, s[44:45]
	v_add_co_u32 v5, s[44:45], s42, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s43, 0, s[44:45]
	v_add_co_u32 v7, s[42:43], s40, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s41, 0, s[42:43]
	global_load_b64 v[110:111], v[1:2], off offset:40
	global_load_b64 v[108:109], v[3:4], off offset:40
	global_load_b64 v[106:107], v[5:6], off offset:40
	global_load_b64 v[104:105], v[7:8], off offset:40
	v_add_nc_u32_e32 v76, s37, v187
	ds_load_2addr_stride64_b32 v[1:2], v78 offset1:2
	ds_load_2addr_stride64_b32 v[3:4], v76 offset1:2
	ds_load_2addr_stride64_b32 v[72:73], v76 offset0:4 offset1:6
	ds_load_2addr_stride64_b32 v[74:75], v78 offset0:4 offset1:6
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v1, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v1, v4, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v1, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v1, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v2, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v2, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v2, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v2, v73, 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v74, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v74, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v74, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v74, v73, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v75, v3, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v75, v4, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v75, v72, 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v75, v73, 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v78 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[74:75], v76 offset0:1 offset1:3
	ds_load_2addr_stride64_b32 v[76:77], v76 offset0:5 offset1:7
	ds_load_2addr_stride64_b32 v[78:79], v78 offset0:5 offset1:7
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v74, 0x2000, v182
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v110, s[0:1]
	v_add_nc_u32_e32 v73, 0x4000, v182
	v_cndmask_b32_e64 v75, 0, v111, s[0:1]
	s_wait_loadcnt 0x2
	ds_store_2addr_b32 v74, v108, v109 offset1:16
	ds_store_2addr_b32 v73, v72, v75 offset1:16
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v106, s[2:3]
	v_cndmask_b32_e64 v73, 0, v107, s[2:3]
	v_add_nc_u32_e32 v74, 0x4000, v185
	v_add_nc_u32_e32 v75, 0x2000, v185
	ds_store_2addr_b32 v74, v72, v73 offset1:16
	s_wait_loadcnt 0x0
	ds_store_2addr_b32 v75, v104, v105 offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b64 s[36:37], s[34:35], s[26:27]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[36:37]
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_15
; %bb.14:                               ;   in Loop: Header=BB3_13 Depth=2
	s_add_co_i32 s22, s22, 1
	s_xor_b32 s46, s38, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[22:23], s[10:11]
	s_add_co_i32 s22, s38, s24
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[40:41], s[40:41], 0x48
	s_mul_u64 s[42:43], s[22:23], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[40:41], s[14:15], s[40:41]
	s_add_nc_u64 s[38:39], s[12:13], s[42:43]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v186, s[40:41]
	v_add_co_u32 v72, s[44:45], s40, v68
	s_lshl_b32 s42, s46, 6
	s_mov_b32 s43, s23
	s_mulk_i32 s22, 0x88
	v_add_co_ci_u32_e64 v73, null, s41, 0, s[44:45]
	v_add_co_u32 v76, vcc, v76, v193
	v_add_co_u32 v74, s[44:45], s40, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[38:39], s[42:43]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc
	v_add_co_u32 v82, vcc, v65, s22
	v_add_co_ci_u32_e64 v75, null, s41, 0, s[44:45]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s[40:41], s38, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s39, 0, s[40:41]
	v_add_co_u32 v80, s[40:41], s38, v71
	s_lshl_b32 s22, s46, 2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s39, 0, s[40:41]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc, v82, s22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc
	s_clause 0x1
	global_load_b64 v[110:111], v[72:73], off offset:8
	global_load_b64 v[106:107], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[108:109], v[78:79], off offset:8
	global_load_b64 v[104:105], v[80:81], off offset:8
	global_load_b32 v194, v[76:77], off
	global_load_b32 v195, v[82:83], off
.LBB3_15:                               ; %.preheader527.i.i
                                        ;   in Loop: Header=BB3_13 Depth=2
	v_add_nc_u32_e32 v80, 0, v188
	v_add_nc_u32_e32 v81, 0, v187
	s_xor_b64 s[36:37], s[36:37], -1
	s_and_b64 s[38:39], s[30:31], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:32 offset1:34
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:64 offset1:66
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:68 offset1:70
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:36 offset1:38
	s_cselect_b32 s38, s21, 0x3c00
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_stride64_b32 v[72:73], v80 offset0:33 offset1:35
	ds_load_2addr_stride64_b32 v[74:75], v81 offset0:65 offset1:67
	ds_load_2addr_stride64_b32 v[76:77], v81 offset0:69 offset1:71
	ds_load_2addr_stride64_b32 v[78:79], v80 offset0:37 offset1:39
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[61:64], v72, v74, v[61:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:60], v72, v75, v[57:60] neg_lo:[0,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[53:56], v72, v76, v[53:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:52], v72, v77, v[49:52] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[45:48], v73, v74, v[45:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:44], v73, v75, v[41:44] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[37:40], v73, v76, v[37:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:36], v73, v77, v[33:36] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[29:32], v78, v74, v[29:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:28], v78, v75, v[25:28] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[21:24], v78, v76, v[21:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:20], v78, v77, v[17:20] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[13:16], v79, v74, v[13:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:12], v79, v75, v[9:12] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[5:8], v79, v76, v[5:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:4], v79, v77, v[1:4] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s38, v191
	v_add_nc_u32_e32 v72, s22, v192
	s_and_not1_b64 vcc, exec, s[36:37]
	s_movk_i32 s36, 0x2000
	s_movk_i32 s37, 0x4000
	ds_load_2addr_b32 v[116:117], v76 offset1:1
	ds_load_2addr_b32 v[118:119], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[114:115], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[112:113], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[86:87], v72 offset1:1
	ds_load_2addr_b32 v[84:85], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[82:83], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[100:101], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[102:103], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[96:97], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[98:99], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[94:95], v76 offset0:64 offset1:65
	ds_load_2addr_b32 v[92:93], v76 offset0:66 offset1:67
	ds_load_2addr_b32 v[90:91], v76 offset0:68 offset1:69
	ds_load_2addr_b32 v[88:89], v76 offset0:70 offset1:71
	ds_load_2addr_b32 v[80:81], v76 offset0:96 offset1:97
	ds_load_2addr_b32 v[78:79], v76 offset0:98 offset1:99
	ds_load_2addr_b32 v[74:75], v76 offset0:100 offset1:101
	ds_load_2addr_b32 v[76:77], v76 offset0:102 offset1:103
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.16:                               ; %.preheader528.i.i
                                        ;   in Loop: Header=BB3_13 Depth=2
	s_wait_loadcnt 0x5
	v_cndmask_b32_e64 v110, 0, v110, s[0:1]
	v_cndmask_b32_e64 v111, 0, v111, s[0:1]
	v_add_nc_u32_e32 v196, 0x1000, v182
	s_and_b64 s[34:35], s[34:35], exec
	s_cselect_b32 s22, s9, 0x3400
	ds_store_2addr_b32 v182, v110, v111 offset1:16
	s_wait_loadcnt 0x3
	ds_store_2addr_b32 v196, v108, v109 offset1:16
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v108, v190, v195
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v111, s22, v189
	s_cselect_b32 s22, s21, 0x3c00
	v_cndmask_b32_e64 v106, 0, v106, s[2:3]
	v_cndmask_b32_e64 v107, 0, v107, s[2:3]
	v_cvt_f32_f16_e32 v108, v108.l
	v_cndmask_b32_e64 v110, 0, v194, s[4:5]
	v_add_nc_u32_e32 v109, 0x1000, v185
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v196, s22, v189
	s_mov_b32 s37, 0
	v_cndmask_b32_e64 v108, 0, v108, s[6:7]
	s_movk_i32 s36, 0x1000
	ds_store_2addr_b32 v185, v106, v107 offset1:16
	ds_store_2addr_b32 v109, v104, v105 offset1:16
	ds_store_b32 v111, v110
	ds_store_b32 v196, v108
	s_branch .LBB3_12
.LBB3_17:                               ; %Flow1133
	v_mov_b32_e32 v5, v179
.LBB3_18:                               ; %._crit_edge599.i.i
	v_lshrrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v3, 15, v0
	v_lshrrev_b32_e32 v6, 4, v0
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s9, s8, 31
	v_mul_u32_u24_e32 v1, 0x500, v1
	v_lshlrev_b32_e32 v2, 2, v3
	v_mul_u32_u24_e32 v0, 0x50, v3
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[8:9], s[18:19]
	s_ashr_i32 s21, s20, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v8, 0, v1, v2
	s_lshl_b64 s[2:3], s[20:21], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	v_mul_lo_u32 v7, s8, v6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mad_i32_i24 v1, 0x50, v5, v8
	s_add_co_i32 s0, s20, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s0, s8
	ds_store_2addr_b32 v1, v183, v184 offset1:20
	ds_store_2addr_b32 v1, v180, v181 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_lshlrev_b32_e32 v1, 2, v6
	s_cselect_b64 s[0:1], -1, 0
	s_add_co_i32 s2, s18, 0x80
	v_or_b32_e32 v6, s18, v6
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s2, s10
	v_add3_u32 v2, 0, v0, v1
	v_or_b32_e32 v0, s20, v3
	s_cselect_b64 s[2:3], -1, 0
	s_mov_b32 s7, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 s[12:13], s[0:1], s[2:3]
	v_cmp_gt_i32_e64 s[2:3], s10, v6
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_mov_b32 s6, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 vcc, exec, s[12:13]
	s_mov_b64 s[14:15], -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_22
; %bb.19:
	s_and_b64 s[2:3], s[0:1], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_21
; %bb.20:
	v_mad_co_i64_i32 v[9:10], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc, v1, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v4, off
.LBB3_21:                               ; %Flow1129
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[14:15], 0
.LBB3_22:                               ; %Flow1130
	v_add_lshl_u32 v3, v7, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_24
; %bb.23:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v3, s[4:7], null offen
.LBB3_24:
	ds_load_b32 v9, v2 offset:1280
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, 0, 1, s[12:13]
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v7, 64, v6
	s_and_not1_b64 vcc, exec, s[12:13]
	s_mov_b64 s[2:3], -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_28
; %bb.25:
	v_cmp_gt_i32_e32 vcc, s10, v7
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off
.LBB3_27:                               ; %Flow1127
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_28:                               ; %Flow1128
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_30
; %bb.29:
	s_lshl_b32 s0, s8, 8
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s0 offen
.LBB3_30:
	s_wait_dscnt 0x0
	ds_load_b32 v9, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 64, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_34
; %bb.31:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_33
; %bb.32:
	v_mad_co_i64_i32 v[10:11], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off offset:256
.LBB3_33:                               ; %Flow1125
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_34:                               ; %Flow1126
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_36
; %bb.35:
	s_movk_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s0 offen
.LBB3_36:
	s_wait_dscnt 0x0
	ds_load_b32 v9, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_40
; %bb.37:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_39
; %bb.38:
	v_mad_co_i64_i32 v[10:11], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off offset:256
.LBB3_39:                               ; %Flow1123
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_40:                               ; %Flow1124
	v_mul_i32_i24_e32 v1, 0x50, v5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_42
; %bb.41:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s0 offen
.LBB3_42:                               ; %.preheader.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v8, v1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v8, 16, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v177, v178 offset1:20
	ds_store_2addr_b32 v5, v175, v176 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v9, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_46
; %bb.43:
	v_cmp_gt_i32_e32 vcc, s10, v8
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_45
; %bb.44:
	v_mad_co_i64_i32 v[10:11], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off
.LBB3_45:                               ; %Flow1121
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_46:                               ; %Flow1122
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_48
; %bb.47:
	s_lshl_b32 s0, s8, 6
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s0 offen
.LBB3_48:
	ds_load_b32 v10, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 0x50, v6
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_52
; %bb.49:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v9
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_51
; %bb.50:
	v_mad_co_i64_i32 v[11:12], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v10, off
.LBB3_51:                               ; %Flow1119
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_52:                               ; %Flow1120
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s9, s8, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_54
; %bb.53:
	s_wait_dscnt 0x0
	buffer_store_b32 v10, v3, s[4:7], s9 offen
.LBB3_54:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_58
; %bb.55:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_57
; %bb.56:
	v_mad_co_i64_i32 v[11:12], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v10, off offset:256
.LBB3_57:                               ; %Flow1117
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_58:                               ; %Flow1118
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_60
; %bb.59:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v10, v3, s[4:7], s0 offen
.LBB3_60:
	s_wait_dscnt 0x0
	ds_load_b32 v10, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_64
; %bb.61:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_63
; %bb.62:
	v_mad_co_i64_i32 v[11:12], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v16, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v10, off offset:256
.LBB3_63:                               ; %Flow1115
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_64:                               ; %Flow1116
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_66
; %bb.65:
	s_add_co_i32 s0, s9, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v10, v3, s[4:7], s0 offen
.LBB3_66:                               ; %.preheader.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v10, 32, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v173, v174 offset1:20
	ds_store_2addr_b32 v5, v171, v172 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v11, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_70
; %bb.67:
	v_cmp_gt_i32_e32 vcc, s10, v10
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_69
; %bb.68:
	v_mad_co_i64_i32 v[12:13], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc, v1, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v16, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v11, off
.LBB3_69:                               ; %Flow1113
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_70:                               ; %Flow1114
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_72
; %bb.71:
	s_lshl_b32 s0, s8, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v11, v3, s[4:7], s0 offen
.LBB3_72:
	ds_load_b32 v12, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 0x60, v6
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_76
; %bb.73:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s10, v11
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_75
; %bb.74:
	v_mad_co_i64_i32 v[15:16], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v12, off
.LBB3_75:                               ; %Flow1111
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_76:                               ; %Flow1112
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s11, s8, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_78
; %bb.77:
	s_wait_dscnt 0x0
	buffer_store_b32 v12, v3, s[4:7], s11 offen
.LBB3_78:
	s_wait_dscnt 0x0
	ds_load_b32 v12, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_82
; %bb.79:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_81
; %bb.80:
	v_mad_co_i64_i32 v[15:16], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v12, off offset:256
.LBB3_81:                               ; %Flow1109
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_82:                               ; %Flow1110
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_84
; %bb.83:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v12, v3, s[4:7], s0 offen
.LBB3_84:
	s_wait_dscnt 0x0
	ds_load_b32 v12, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_88
; %bb.85:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_87
; %bb.86:
	v_mad_co_i64_i32 v[15:16], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v13, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v12, off offset:256
.LBB3_87:                               ; %Flow1107
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_88:                               ; %Flow1108
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_90
; %bb.89:
	s_add_co_i32 s0, s11, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v12, v3, s[4:7], s0 offen
.LBB3_90:                               ; %.preheader.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s8, v0
	v_or_b32_e32 v12, 48, v6
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v169, v170 offset1:20
	ds_store_2addr_b32 v5, v167, v168 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_94
; %bb.91:
	v_cmp_gt_i32_e32 vcc, s10, v12
	s_and_b64 s[2:3], s[0:1], vcc
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_93
; %bb.92:
	v_mad_co_i64_i32 v[15:16], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v13, off
.LBB3_93:                               ; %Flow1105
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_94:                               ; %Flow1106
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_mul_i32 s12, s8, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_96
; %bb.95:
	s_wait_dscnt 0x0
	buffer_store_b32 v13, v3, s[4:7], s12 offen
.LBB3_96:
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v13, 0x70, v6
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_100
; %bb.97:
	v_cmp_gt_i32_e32 vcc, s8, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_99
; %bb.98:
	v_mad_co_i64_i32 v[16:17], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off
.LBB3_99:                               ; %Flow1103
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_100:                              ; %Flow1104
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_mul_i32 s13, s8, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_102
; %bb.101:
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s13 offen
.LBB3_102:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_106
; %bb.103:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_105
; %bb.104:
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:256
.LBB3_105:                              ; %Flow1101
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_106:                              ; %Flow1102
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_108
; %bb.107:
	s_add_co_i32 s0, s12, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_108:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_112
; %bb.109:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_111
; %bb.110:
	v_mad_co_i64_i32 v[16:17], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v14, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:256
.LBB3_111:                              ; %Flow1099
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_112:                              ; %Flow1100
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_114
; %bb.113:
	s_add_co_i32 s0, s13, 0x100
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_114:                              ; %.preheader524.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 16, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v165, v166 offset1:20
	ds_store_2addr_b32 v5, v163, v164 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_118
; %bb.115:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_117
; %bb.116:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:64
.LBB3_117:                              ; %Flow1097
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_118:                              ; %Flow1098
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_120
; %bb.119:
	s_mov_b32 s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_120:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_124
; %bb.121:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_123
; %bb.122:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:64
.LBB3_123:                              ; %Flow1095
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_124:                              ; %Flow1096
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_126
; %bb.125:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_126:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x50, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_130
; %bb.127:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_129
; %bb.128:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_129:                              ; %Flow1093
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_130:                              ; %Flow1094
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_132
; %bb.131:
	s_movk_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_132:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_136
; %bb.133:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_135
; %bb.134:
	v_mad_co_i64_i32 v[17:18], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_135:                              ; %Flow1091
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_136:                              ; %Flow1092
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_138
; %bb.137:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_138:                              ; %.preheader.1.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v161, v162 offset1:20
	ds_store_2addr_b32 v5, v159, v160 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_142
; %bb.139:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_141
; %bb.140:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_141:                              ; %Flow1089
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_142:                              ; %Flow1090
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_144
; %bb.143:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_144:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_148
; %bb.145:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_147
; %bb.146:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_147:                              ; %Flow1087
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_148:                              ; %Flow1088
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_150
; %bb.149:
	s_add_co_i32 s0, s9, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_150:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_154
; %bb.151:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_153
; %bb.152:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_153:                              ; %Flow1085
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_154:                              ; %Flow1086
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_156
; %bb.155:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_156:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_160
; %bb.157:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_159
; %bb.158:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_159:                              ; %Flow1083
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_160:                              ; %Flow1084
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_162
; %bb.161:
	s_add_co_i32 s0, s9, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_162:                              ; %.preheader.2.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v157, v158 offset1:20
	ds_store_2addr_b32 v5, v155, v156 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_166
; %bb.163:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_165
; %bb.164:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_165:                              ; %Flow1081
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_166:                              ; %Flow1082
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_168
; %bb.167:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, s0, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_168:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_172
; %bb.169:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_171
; %bb.170:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_171:                              ; %Flow1079
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_172:                              ; %Flow1080
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_174
; %bb.173:
	s_or_b32 s0, s11, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_174:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_178
; %bb.175:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_177
; %bb.176:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_177:                              ; %Flow1077
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_178:                              ; %Flow1078
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_180
; %bb.179:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_180:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_184
; %bb.181:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_183
; %bb.182:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:320
.LBB3_183:                              ; %Flow1075
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_184:                              ; %Flow1076
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_186
; %bb.185:
	s_add_co_i32 s0, s11, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_186:                              ; %.preheader.3.1.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v153, v154 offset1:20
	ds_store_2addr_b32 v5, v151, v152 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_190
; %bb.187:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_189
; %bb.188:
	v_mad_co_i64_i32 v[17:18], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_189:                              ; %Flow1073
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_190:                              ; %Flow1074
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_192
; %bb.191:
	s_add_co_i32 s0, s12, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_192:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_196
; %bb.193:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_195
; %bb.194:
	v_mad_co_i64_i32 v[17:18], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v14, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:64
.LBB3_195:                              ; %Flow1071
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_196:                              ; %Flow1072
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_198
; %bb.197:
	s_add_co_i32 s0, s13, 64
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_198:
	ds_load_b32 v14, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_202
; %bb.199:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_201
; %bb.200:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v14, off offset:320
.LBB3_201:                              ; %Flow1069
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_202:                              ; %Flow1070
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_204
; %bb.203:
	s_add_co_i32 s0, s12, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s0 offen
.LBB3_204:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_208
; %bb.205:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_207
; %bb.206:
	v_mad_co_i64_i32 v[15:16], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off offset:320
.LBB3_207:                              ; %Flow1067
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_208:                              ; %Flow1068
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_210
; %bb.209:
	s_add_co_i32 s0, s13, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s0 offen
.LBB3_210:                              ; %.preheader524.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 32, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v149, v150 offset1:20
	ds_store_2addr_b32 v5, v147, v148 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_214
; %bb.211:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_213
; %bb.212:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:128
.LBB3_213:                              ; %Flow1065
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_214:                              ; %Flow1066
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_216
; %bb.215:
	s_movk_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_216:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_220
; %bb.217:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_219
; %bb.218:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:128
.LBB3_219:                              ; %Flow1063
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_220:                              ; %Flow1064
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_222
; %bb.221:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_bitset1_b32 s0, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_222:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x60, v0
	s_mov_b64 s[2:3], -1
	s_cbranch_vccnz .LBB3_226
; %bb.223:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_225
; %bb.224:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_225:                              ; %Flow1061
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_226:                              ; %Flow1062
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_228
; %bb.227:
	s_movk_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_228:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_232
; %bb.229:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_231
; %bb.230:
	v_mad_co_i64_i32 v[17:18], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_231:                              ; %Flow1059
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_232:                              ; %Flow1060
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_234
; %bb.233:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_234:                              ; %.preheader.1.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v145, v146 offset1:20
	ds_store_2addr_b32 v5, v143, v144 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_238
; %bb.235:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_237
; %bb.236:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_237:                              ; %Flow1057
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_238:                              ; %Flow1058
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_240
; %bb.239:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_240:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_244
; %bb.241:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_243
; %bb.242:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_243:                              ; %Flow1055
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_244:                              ; %Flow1056
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_246
; %bb.245:
	s_add_co_i32 s0, s9, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_246:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_250
; %bb.247:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_249
; %bb.248:
	v_mad_co_i64_i32 v[17:18], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_249:                              ; %Flow1053
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_250:                              ; %Flow1054
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_252
; %bb.251:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_252:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_256
; %bb.253:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_255
; %bb.254:
	v_mad_co_i64_i32 v[17:18], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_255:                              ; %Flow1051
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_256:                              ; %Flow1052
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_258
; %bb.257:
	s_add_co_i32 s0, s9, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_258:                              ; %.preheader.2.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v141, v142 offset1:20
	ds_store_2addr_b32 v5, v139, v140 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_262
; %bb.259:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_261
; %bb.260:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_261:                              ; %Flow1049
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_262:                              ; %Flow1050
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_264
; %bb.263:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_264:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_268
; %bb.265:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_267
; %bb.266:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_267:                              ; %Flow1047
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_268:                              ; %Flow1048
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_270
; %bb.269:
	s_add_co_i32 s0, s11, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_270:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_274
; %bb.271:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_273
; %bb.272:
	v_mad_co_i64_i32 v[17:18], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_273:                              ; %Flow1045
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_274:                              ; %Flow1046
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_276
; %bb.275:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_276:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_280
; %bb.277:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_279
; %bb.278:
	v_mad_co_i64_i32 v[17:18], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:384
.LBB3_279:                              ; %Flow1043
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_280:                              ; %Flow1044
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_282
; %bb.281:
	s_add_co_i32 s0, s11, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_282:                              ; %.preheader.3.2.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v137, v138 offset1:20
	ds_store_2addr_b32 v5, v135, v136 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v16, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_286
; %bb.283:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_285
; %bb.284:
	v_mad_co_i64_i32 v[17:18], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_285:                              ; %Flow1041
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_286:                              ; %Flow1042
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_288
; %bb.287:
	s_add_co_i32 s0, s12, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_288:
	s_wait_dscnt 0x0
	ds_load_b32 v16, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_292
; %bb.289:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_291
; %bb.290:
	v_mad_co_i64_i32 v[17:18], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v14, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:128
.LBB3_291:                              ; %Flow1039
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_292:                              ; %Flow1040
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_294
; %bb.293:
	s_add_co_i32 s0, s13, 0x80
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_294:
	ds_load_b32 v14, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_298
; %bb.295:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_297
; %bb.296:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v14, off offset:384
.LBB3_297:                              ; %Flow1037
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_298:                              ; %Flow1038
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_300
; %bb.299:
	s_add_co_i32 s0, s12, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s0 offen
.LBB3_300:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_304
; %bb.301:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_303
; %bb.302:
	v_mad_co_i64_i32 v[15:16], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s17, v16, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, vcc, v1, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off offset:384
.LBB3_303:                              ; %Flow1035
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_304:                              ; %Flow1036
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_306
; %bb.305:
	s_add_co_i32 s0, s13, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s0 offen
.LBB3_306:                              ; %.preheader524.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	v_or_b32_e32 v14, 48, v0
	s_mov_b64 s[2:3], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v133, v134 offset1:20
	ds_store_2addr_b32 v5, v131, v132 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_310
; %bb.307:
	v_cmp_gt_i32_e32 vcc, s8, v14
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_309
; %bb.308:
	v_mad_co_i64_i32 v[16:17], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:192
.LBB3_309:                              ; %Flow1033
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[2:3], 0
.LBB3_310:                              ; %Flow1034
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_312
; %bb.311:
	s_movk_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_312:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_316
; %bb.313:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_315
; %bb.314:
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off offset:192
.LBB3_315:                              ; %Flow1031
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_316:                              ; %Flow1032
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_318
; %bb.317:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s0 offen
.LBB3_318:
	ds_load_b32 v16, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v15, 0x70, v0
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_322
; %bb.319:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_321
; %bb.320:
	v_mad_co_i64_i32 v[17:18], null, s8, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[19:20], 2, v[0:1]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v18, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc, v1, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v6, v20, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v16, off offset:448
.LBB3_321:                              ; %Flow1029
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_322:                              ; %Flow1030
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_324
; %bb.323:
	s_movk_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v16, v3, s[4:7], s0 offen
.LBB3_324:
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_328
; %bb.325:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_327
; %bb.326:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[16:17], null, s8, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v6, off offset:448
.LBB3_327:                              ; %Flow1027
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_328:                              ; %Flow1028
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_330
; %bb.329:
	s_lshl_b32 s0, s8, 8
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_330:                              ; %.preheader.1.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v129, v130 offset1:20
	ds_store_2addr_b32 v5, v127, v128 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_334
; %bb.331:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_333
; %bb.332:
	v_mad_co_i64_i32 v[16:17], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v6, off offset:192
.LBB3_333:                              ; %Flow1025
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_334:                              ; %Flow1026
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_336
; %bb.335:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_336:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_340
; %bb.337:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_339
; %bb.338:
	v_mad_co_i64_i32 v[16:17], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v17, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc, v1, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v7, v19, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v6, off offset:192
.LBB3_339:                              ; %Flow1023
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_340:                              ; %Flow1024
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_342
; %bb.341:
	s_add_co_i32 s0, s9, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_342:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_346
; %bb.343:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_345
; %bb.344:
	v_mad_co_i64_i32 v[7:8], null, s8, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:448
.LBB3_345:                              ; %Flow1021
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_346:                              ; %Flow1022
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_348
; %bb.347:
	s_lshl_b32 s0, s8, 6
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_348:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_352
; %bb.349:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_351
; %bb.350:
	v_mad_co_i64_i32 v[7:8], null, s8, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:448
.LBB3_351:                              ; %Flow1019
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_352:                              ; %Flow1020
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_354
; %bb.353:
	s_addk_co_i32 s9, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s9 offen
.LBB3_354:                              ; %.preheader.2.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v125, v126 offset1:20
	ds_store_2addr_b32 v5, v124, v123 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v6, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_358
; %bb.355:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_357
; %bb.356:
	v_mad_co_i64_i32 v[7:8], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:192
.LBB3_357:                              ; %Flow1017
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_358:                              ; %Flow1018
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_360
; %bb.359:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_360:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_364
; %bb.361:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_363
; %bb.362:
	v_mad_co_i64_i32 v[7:8], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v17, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:192
.LBB3_363:                              ; %Flow1015
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_364:                              ; %Flow1016
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_366
; %bb.365:
	s_add_co_i32 s0, s11, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_366:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_370
; %bb.367:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_369
; %bb.368:
	v_mad_co_i64_i32 v[7:8], null, s8, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:448
.LBB3_369:                              ; %Flow1013
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_370:                              ; %Flow1014
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_372
; %bb.371:
	s_lshl_b32 s0, s8, 7
	s_wait_alu depctr_sa_sdst(0)
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s0 offen
.LBB3_372:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_376
; %bb.373:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_375
; %bb.374:
	v_mad_co_i64_i32 v[7:8], null, s8, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v6, off offset:448
.LBB3_375:                              ; %Flow1011
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_376:                              ; %Flow1012
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_378
; %bb.377:
	s_addk_co_i32 s11, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v6, v3, s[4:7], s11 offen
.LBB3_378:                              ; %.preheader.3.3.i.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_and_b64 vcc, exec, vcc
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v120, v122 offset1:20
	ds_store_2addr_b32 v5, v67, v121 offset0:40 offset1:60
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v5, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_382
; %bb.379:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_381
; %bb.380:
	v_mad_co_i64_i32 v[6:7], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off offset:192
.LBB3_381:                              ; %Flow1009
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_382:                              ; %Flow1010
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_384
; %bb.383:
	s_add_co_i32 s0, s12, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s0 offen
.LBB3_384:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:1280
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_388
; %bb.385:
	v_cmp_gt_i32_e32 vcc, s8, v14
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_387
; %bb.386:
	v_mad_co_i64_i32 v[6:7], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off offset:192
.LBB3_387:                              ; %Flow1007
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_388:                              ; %Flow1008
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_390
; %bb.389:
	s_add_co_i32 s0, s13, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s0 offen
.LBB3_390:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:2560
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_394
; %bb.391:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_393
; %bb.392:
	v_mad_co_i64_i32 v[6:7], null, s8, v12, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc, s16, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off offset:448
.LBB3_393:                              ; %Flow1005
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_394:                              ; %Flow1006
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_396
; %bb.395:
	s_addk_co_i32 s12, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s12 offen
.LBB3_396:
	ds_load_b32 v2, v2 offset:3840
	v_cmp_ne_u32_e32 vcc, 1, v4
	s_mov_b64 s[0:1], -1
	s_cbranch_vccnz .LBB3_400
; %bb.397:
	v_cmp_gt_i32_e32 vcc, s8, v15
	v_cmp_gt_i32_e64 s[0:1], s10, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_b64 s[2:3], vcc, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB3_399
; %bb.398:
	s_wait_dscnt 0x1
	v_mad_co_i64_i32 v[4:5], null, s8, v13, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v2, off offset:448
.LBB3_399:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB3_400:                              ; %Flow1004
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b64 vcc, exec, s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_402
; %bb.401:
	s_addk_co_i32 s13, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v3, s[4:7], s13 offen
.LBB3_402:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_403:                              ; %_ZL40gemm_mq4g256v2_residual_mmq_iu4_w64_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.Lfunc_end3:
	.size	iu4_w64_r4_set, .Lfunc_end3-iu4_w64_r4_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel iu4_w64_r4_set
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
		.amdhsa_next_free_vgpr 197
		.amdhsa_next_free_sgpr 47
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-iu4_w64_r4_set)<<4)&4080)>>4
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
	.set .Liu4_w64_r4_set.num_vgpr, 197
	.set .Liu4_w64_r4_set.num_agpr, 0
	.set .Liu4_w64_r4_set.numbered_sgpr, 47
	.set .Liu4_w64_r4_set.num_named_barrier, 0
	.set .Liu4_w64_r4_set.private_seg_size, 0
	.set .Liu4_w64_r4_set.uses_vcc, 1
	.set .Liu4_w64_r4_set.uses_flat_scratch, 0
	.set .Liu4_w64_r4_set.has_dyn_sized_stack, 0
	.set .Liu4_w64_r4_set.has_recursion, 0
	.set .Liu4_w64_r4_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18848
; TotalNumSgprs: 49
; NumVgprs: 197
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 49
; NumSGPRsForWavesPerEU: 49
; NumVGPRsForWavesPerEU: 197
; Occupancy: 3
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
	.type	__hip_cuid_920766ad8129eabc,@object ; @__hip_cuid_920766ad8129eabc
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_920766ad8129eabc
__hip_cuid_920766ad8129eabc:
	.byte	0                               ; 0x0
	.size	__hip_cuid_920766ad8129eabc, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_920766ad8129eabc
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
    .name:           iu4_w64_r4
    .private_segment_fixed_size: 0
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     200
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
    .name:           iu4_w64_r4_add
    .private_segment_fixed_size: 0
    .sgpr_count:     49
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r4_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     197
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
    .name:           iu4_w64_r4_set
    .private_segment_fixed_size: 0
    .sgpr_count:     49
    .sgpr_spill_count: 0
    .symbol:         iu4_w64_r4_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     197
    .vgpr_spill_count: 0
    .wavefront_size: 64
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
