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
	s_load_b32 s4, s[0:1], 0x24
	s_load_b64 s[2:3], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_lt_i32 ttmp7, s3
	v_mad_co_u64_u32 v[5:6], null, ttmp9, s4, v[0:1]
	s_cselect_b32 s4, -1, 0
	v_lshlrev_b32_e32 v6, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s2, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB0_16
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_or_b32_e32 v1, 3, v6
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s8, ttmp7
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, s2, v1
	v_mov_b32_e32 v1, 0
	s_ashr_i32 s9, ttmp7, 31
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_ashrrev_i32_e32 v7, 31, v6
	s_ashr_i32 s11, s2, 31
	s_mov_b32 s10, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mul_u64 s[10:11], s[10:11], s[8:9]
	v_lshlrev_b64_e32 v[1:2], 2, v[6:7]
	s_lshl_b64 s[10:11], s[10:11], 2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[4:5], s[4:5], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s4, v1
	v_add_co_ci_u32_e64 v2, null, s5, v2, vcc_lo
	global_load_b128 v[1:4], v[1:2], off
.LBB0_3:                                ; %._crit_edge
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_wait_loadcnt 0x0
	v_max3_num_f32 v8, |v1|, |v2|, |v3|
	v_max_num_f32_e64 v9, |v4|, |v4|
	s_wait_kmcnt 0x0
	s_mov_b32 s5, 0
	s_mov_b32 s4, exec_lo
	v_xor_b32_e32 v7, 16, v6
	v_xor_b32_e32 v10, 8, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v6, v7, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v10
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v10, v6, v10 :: v_dual_max_num_f32 v9, v8, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 2, v7
	ds_bpermute_b32 v8, v7, v9
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v11, v8, v8 :: v_dual_lshlrev_b32 v8, 2, v10
	v_max_num_f32_e32 v10, v9, v11
	v_xor_b32_e32 v11, 4, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo
	ds_bpermute_b32 v9, v8, v10
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v12, v9, v9 :: v_dual_lshlrev_b32 v9, 2, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_max_num_f32_e32 v10, v10, v12
	v_xor_b32_e32 v12, 2, v6
	ds_bpermute_b32 v11, v9, v10
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v12, v6, v12 :: v_dual_max_num_f32 v13, v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v12, v10, v13 :: v_dual_lshlrev_b32 v11, 2, v12
	v_xor_b32_e32 v13, 1, v6
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v6, v13, vcc_lo
	ds_bpermute_b32 v10, v11, v12
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v13, v10, v10
	v_lshlrev_b32_e32 v10, 2, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v6, v12, v13
	ds_bpermute_b32 v12, v10, v6
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v12, v6, v12
	v_mov_b32_e32 v6, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_6
; %bb.4:                                ; %.preheader68.i.i
	v_div_scale_f32 v6, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s9, 0x40e00000
	s_mov_b32 s10, 0xc1000000
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
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v6, 1.0 :: v_dual_mul_f32 v13, 0.5, v13
.LBB0_5:                                ; %.preheader.preheader.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s0, s5
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 8
	v_div_scale_f32 v15, null, s9, s9, s0
	v_div_scale_f32 v16, vcc_lo, s0, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_2)
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
	v_div_scale_f32 v17, vcc_lo, v1, v15, v1
	v_rcp_f32_e32 v24, v16
	v_rcp_f32_e32 v25, v18
	v_rcp_f32_e32 v26, v20
	v_rcp_f32_e32 v27, v22
	v_div_scale_f32 v19, s0, v2, v15, v2
	v_div_scale_f32 v21, s1, v3, v15, v3
	v_div_scale_f32 v23, s2, v4, v15, v4
	v_fma_f32 v28, -v16, v24, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v29, -v18, v25, 1.0
	v_fma_f32 v30, -v20, v26, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v31, -v22, v27, 1.0
	v_dual_fmac_f32 v24, v28, v24 :: v_dual_fmac_f32 v25, v29, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v26, v30, v26 :: v_dual_fmac_f32 v27, v31, v27
	v_dual_mul_f32 v28, v17, v24 :: v_dual_mul_f32 v29, v19, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v30, v21, v26 :: v_dual_mul_f32 v31, v23, v27
	v_fma_f32 v32, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v33, -v18, v29, v19
	v_fma_f32 v34, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v35, -v22, v31, v23
	v_dual_fmac_f32 v28, v32, v24 :: v_dual_fmac_f32 v29, v33, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v30, v34, v26 :: v_dual_fmac_f32 v31, v35, v27
	v_fma_f32 v16, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v29, v19
	v_fma_f32 v18, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v19, -v22, v31, v23
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v24, v28
	s_mov_b32 vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v25, v29
	s_mov_b32 vcc_lo, s1
	v_div_fixup_f32 v16, v16, v15, v1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v26, v30
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v19, v27, v31
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v3
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v19, v19, v15, v4
	v_med3_num_f32 v16, v16, s10, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s10, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v19, v19
	v_fma_f32 v16, -v16, v15, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s10, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v19, v19, s10, 0x40e00000
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
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo
	v_cndmask_b32_e32 v6, v6, v15, vcc_lo
	s_cbranch_scc1 .LBB0_5
.LBB0_6:                                ; %Flow39
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_neq_f32_e64 s0, 0, v12
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v12, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_div_scale_f32 v12, null, v6, v6, v1
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v12
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v12, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v12, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v12, -v12, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v12, v12, v14, v16
	v_div_fixup_f32 v1, v12, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v1
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_div_scale_f32 v1, null, v6, v6, v2
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v13, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v2, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v13
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v13
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v13, v15
	v_div_fixup_f32 v1, v1, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v1
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v1, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_div_scale_f32 v1, null, v6, v6, v3
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v2, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v1, v2, 1.0
	v_fmac_f32_e32 v2, v15, v2
	v_div_scale_f32 v15, vcc_lo, v3, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v2
	v_fma_f32 v17, -v1, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v2
	v_fma_f32 v1, -v1, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v2, v16
	v_div_fixup_f32 v1, v1, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v1, v1
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_div_scale_f32 v2, null, v6, v6, v4
	s_mov_b32 s0, 0xc1000000
	v_rcp_f32_e32 v3, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v2, v3, 1.0
	v_fmac_f32_e32 v3, v14, v3
	v_div_scale_f32 v14, vcc_lo, v4, v6, v4
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
	v_cvt_i32_f32_e32 v14, v2
.LBB0_14:                               ; %_Z31quantize_block_i4_128_wave_regsPKfi.exit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v2, v13, v12
	v_ashrrev_i32_e32 v4, 31, v5
	v_and_b32_e32 v0, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add3_u32 v2, v2, v1, v14
	v_lshrrev_b32_e32 v4, 27, v4
	ds_bpermute_b32 v3, v7, v2
	v_add_nc_u32_e32 v4, v5, v4
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v8, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v2, v3
	ds_bpermute_b32 v3, v9, v2
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v5, v2, v3
	v_ashrrev_i32_e32 v2, 5, v4
	ds_bpermute_b32 v4, v11, v5
	v_mad_co_i64_i32 v[2:3], null, v2, s3, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[7:8], null, 0x48, v2, s[6:7]
	v_mad_co_u64_u32 v[8:9], null, 0x48, v3, v[8:9]
	v_lshlrev_b32_e32 v9, 8, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v5, v4
	v_lshlrev_b32_e32 v5, 4, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_i64_i32 v[1:2], null, 0x48, s8, v[7:8]
	ds_bpermute_b32 v4, v10, v3
	v_and_b32_e32 v10, 15, v12
	v_and_b32_e32 v5, 0xf0, v5
	v_and_b32_e32 v7, 0xf00, v9
	v_lshlrev_b32_e32 v9, 1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v8, v14, 12, v10
	v_or3_b32 v5, v8, v5, v7
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v2, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	global_store_b16 v[7:8], v5, off offset:8
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v3, v4
	global_store_b64 v[1:2], v[6:7], off
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
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 12
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
	.set .Lquantize_int4_mmq_ds128.numbered_sgpr, 12
	.set .Lquantize_int4_mmq_ds128.num_named_barrier, 0
	.set .Lquantize_int4_mmq_ds128.private_seg_size, 0
	.set .Lquantize_int4_mmq_ds128.uses_vcc, 1
	.set .Lquantize_int4_mmq_ds128.uses_flat_scratch, 0
	.set .Lquantize_int4_mmq_ds128.has_dyn_sized_stack, 0
	.set .Lquantize_int4_mmq_ds128.has_recursion, 0
	.set .Lquantize_int4_mmq_ds128.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2292
; TotalNumSgprs: 14
; NumVgprs: 36
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 14
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
	.protected	quantize_int4_mmq_ds128_atiled ; -- Begin function quantize_int4_mmq_ds128_atiled
	.globl	quantize_int4_mmq_ds128_atiled
	.p2align	8
	.type	quantize_int4_mmq_ds128_atiled,@function
quantize_int4_mmq_ds128_atiled:         ; @quantize_int4_mmq_ds128_atiled
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x24
	s_load_b64 s[8:9], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_and_b32 s2, s2, 0xffff
	s_cmp_lt_i32 ttmp7, s9
	v_mad_co_u64_u32 v[5:6], null, ttmp9, s2, v[0:1]
	s_cselect_b32 s2, -1, 0
	v_lshlrev_b32_e32 v6, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s8, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s2, s2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_18
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_or_b32_e32 v1, 3, v6
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, s8, v1
	v_mov_b32_e32 v1, 0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB1_3
; %bb.2:
	v_ashrrev_i32_e32 v7, 31, v6
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s2, ttmp7
	s_ashr_i32 s3, ttmp7, 31
	s_ashr_i32 s11, s8, 31
	s_mov_b32 s10, s8
	v_lshlrev_b64_e32 v[1:2], 2, v[6:7]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[10:11], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[2:3], s[2:3], 2
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s3, v2, vcc_lo
	global_load_b128 v[1:4], v[1:2], off
.LBB1_3:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_wait_loadcnt 0x0
	v_max3_num_f32 v8, |v1|, |v2|, |v3|
	v_max_num_f32_e64 v9, |v4|, |v4|
	s_wait_kmcnt 0x0
	s_mov_b32 s4, 0
	s_mov_b32 s3, exec_lo
	v_xor_b32_e32 v7, 16, v6
	v_xor_b32_e32 v11, 4, v6
	v_max_num_f32_e32 v10, v8, v9
	v_xor_b32_e32 v9, 8, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v6, v7, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_cndmask_b32 v9, v6, v9 :: v_dual_lshlrev_b32 v8, 2, v7
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	ds_bpermute_b32 v7, v8, v10
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v7, v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v7, v10, v7
	ds_bpermute_b32 v10, v9, v7
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v10, v10
	v_max_num_f32_e32 v7, v7, v12
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo
	v_xor_b32_e32 v12, 2, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v10, 2, v11
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	ds_bpermute_b32 v11, v10, v7
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v12, v6, v12 :: v_dual_max_num_f32 v13, v11, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v12, v7, v13 :: v_dual_lshlrev_b32 v11, 2, v12
	v_xor_b32_e32 v13, 1, v6
	ds_bpermute_b32 v7, v11, v12
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v6, v6, v13 :: v_dual_max_num_f32 v13, v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v6, v12, v13 :: v_dual_lshlrev_b32 v7, 2, v6
	ds_bpermute_b32 v12, v7, v6
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v12, v6, v12
	v_mov_b32_e32 v6, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB1_6
; %bb.4:                                ; %.preheader68.i
	v_div_scale_f32 v6, null, 0x40e00000, 0x40e00000, v12
	v_div_scale_f32 v15, vcc_lo, v12, 0x40e00000, v12
	s_mov_b32 s5, 0x40e00000
	s_mov_b32 s10, 0xc1000000
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
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v6, 1.0 :: v_dual_mul_f32 v13, 0.5, v13
.LBB1_5:                                ; %.preheader.preheader.i
                                        ; =>This Inner Loop Header: Depth=1
	s_cvt_f32_u32 s0, s4
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 8
	v_div_scale_f32 v15, null, s5, s5, s0
	v_div_scale_f32 v16, vcc_lo, s0, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_2)
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
	v_div_scale_f32 v17, vcc_lo, v1, v15, v1
	v_rcp_f32_e32 v24, v16
	v_rcp_f32_e32 v25, v18
	v_rcp_f32_e32 v26, v20
	v_rcp_f32_e32 v27, v22
	v_div_scale_f32 v19, s0, v2, v15, v2
	v_div_scale_f32 v21, s1, v3, v15, v3
	v_div_scale_f32 v23, s2, v4, v15, v4
	v_fma_f32 v28, -v16, v24, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v29, -v18, v25, 1.0
	v_fma_f32 v30, -v20, v26, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v31, -v22, v27, 1.0
	v_dual_fmac_f32 v24, v28, v24 :: v_dual_fmac_f32 v25, v29, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v26, v30, v26 :: v_dual_fmac_f32 v27, v31, v27
	v_dual_mul_f32 v28, v17, v24 :: v_dual_mul_f32 v29, v19, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v30, v21, v26 :: v_dual_mul_f32 v31, v23, v27
	v_fma_f32 v32, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v33, -v18, v29, v19
	v_fma_f32 v34, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v35, -v22, v31, v23
	v_dual_fmac_f32 v28, v32, v24 :: v_dual_fmac_f32 v29, v33, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v30, v34, v26 :: v_dual_fmac_f32 v31, v35, v27
	v_fma_f32 v16, -v16, v28, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v18, v29, v19
	v_fma_f32 v18, -v20, v30, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fma_f32 v19, -v22, v31, v23
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v16, v16, v24, v28
	s_mov_b32 vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v25, v29
	s_mov_b32 vcc_lo, s1
	v_div_fixup_f32 v16, v16, v15, v1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v26, v30
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v17, v17, v15, v2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v19, v19, v27, v31
	v_rndne_f32_e32 v16, v16
	v_div_fixup_f32 v18, v18, v15, v3
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_fixup_f32 v19, v19, v15, v4
	v_med3_num_f32 v16, v16, s10, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v18, v18
	v_med3_num_f32 v17, v17, s10, 0x40e00000
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_rndne_f32_e32 v19, v19
	v_fma_f32 v16, -v16, v15, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v18, v18, s10, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_med3_num_f32 v19, v19, s10, 0x40e00000
	v_fma_f32 v16, v16, v16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v15, v3
	v_fmac_f32_e32 v16, v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v17, -v19, v15, v4
	v_fmac_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v17
	ds_bpermute_b32 v17, v8, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v9, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v10, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v11, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	ds_bpermute_b32 v17, v7, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo
	v_cndmask_b32_e32 v6, v6, v15, vcc_lo
	s_cbranch_scc1 .LBB1_5
.LBB1_6:                                ; %Flow43
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_cmp_neq_f32_e64 s0, 0, v12
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v12, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_8
; %bb.7:
	v_div_scale_f32 v12, null, v6, v6, v1
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v12
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v12, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v12, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v12, -v12, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v12, v12, v14, v16
	v_div_fixup_f32 v1, v12, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v1
.LBB1_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_10
; %bb.9:
	v_div_scale_f32 v1, null, v6, v6, v2
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v13, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v13, 1.0
	v_fmac_f32_e32 v13, v14, v13
	v_div_scale_f32 v14, vcc_lo, v2, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v13
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v13
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v13, v15
	v_div_fixup_f32 v1, v1, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v1
.LBB1_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_12
; %bb.11:
	v_div_scale_f32 v2, null, v6, v6, v3
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v2, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v3, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v2, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v2, -v2, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v2, v2, v14, v16
	v_div_fixup_f32 v2, v2, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v2
.LBB1_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_14
; %bb.13:
	v_div_scale_f32 v1, null, v6, v6, v4
	s_mov_b32 s0, 0xc1000000
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v14, v3
	v_div_scale_f32 v14, vcc_lo, v4, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v3
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v3
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v15
	v_div_fixup_f32 v1, v1, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s0, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v1, v1
.LBB1_14:                               ; %_Z31quantize_block_i4_128_wave_regsPKfi.exit
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v3, v13, v12
	s_add_co_i32 s0, s9, 0x7f
	s_mov_b32 s2, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s1, s1, 25
	v_add3_u32 v3, v3, v2, v1
	v_lshlrev_b32_e32 v2, 8, v2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s0, s0, 7
	ds_bpermute_b32 v4, v8, v3
	v_lshlrev_b32_e32 v8, 4, v0
	v_and_b32_e32 v2, 0xf00, v2
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v9, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v10, v3
	v_and_b32_e32 v10, 15, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v1, v1, 12, v10
	v_and_b32_e32 v10, 64, v8
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	v_lshlrev_b32_e32 v4, 4, v13
	ds_bpermute_b32 v9, v11, v3
	v_and_b32_e32 v4, 0xf0, v4
	v_and_b32_e32 v11, 3, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or3_b32 v1, v1, v4, v2
	v_lshl_or_b32 v4, v11, 4, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v13, 0xffff, v1
	ds_bpermute_b32 v12, v4, v13
	ds_bpermute_b32 v10, v4, v13 offset:8
	s_wait_dscnt 0x2
	v_add_nc_u32_e32 v1, v3, v9
	v_ashrrev_i32_e32 v3, 31, v5
	ds_bpermute_b32 v9, v4, v13 offset:4
	ds_bpermute_b32 v2, v7, v1
	ds_bpermute_b32 v7, v4, v13 offset:12
	v_lshrrev_b32_e32 v3, 27, v3
	v_and_b32_e32 v4, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v3, v5, v3
	v_ashrrev_i32_e32 v3, 5, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_u32_e32 8, v4
	s_cbranch_execz .LBB1_16
; %bb.15:
	v_bfe_u32 v0, v0, 2, 3
	s_ashr_i32 s4, ttmp7, 7
	s_lshr_b32 s3, ttmp7, 3
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 31
	s_wait_dscnt 0x4
	v_and_b32_e32 v5, 0xffff, v12
	v_lshl_or_b32 v0, v3, 1, v0
	s_wait_dscnt 0x3
	v_and_b32_e32 v10, 0xffff, v10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[13:14], null, v0, s0, s[4:5]
	v_lshrrev_b32_e32 v0, 1, v11
	s_and_b32 s4, ttmp7, 15
	v_and_or_b32 v0, s3, 14, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 12, v[13:14]
	s_wait_alu depctr_sa_sdst(0)
	v_and_or_b32 v13, v8, 16, s4
	s_wait_dscnt 0x2
	v_lshl_or_b32 v8, v9, 16, v5
	v_lshlrev_b32_e32 v0, 8, v0
	v_add_co_u32 v5, vcc_lo, s6, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s7, v12, vcc_lo
	v_lshlrev_b32_e32 v11, 3, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v5, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v9, vcc_lo
	s_wait_dscnt 0x0
	v_lshl_or_b32 v9, v7, 16, v10
	v_add_co_u32 v10, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, 0, v5, vcc_lo
	global_store_b64 v[10:11], v[8:9], off
.LBB1_16:
	s_or_b32 exec_lo, exec_lo, s2
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_18
; %bb.17:
	s_ashr_i32 s3, s8, 31
	s_mov_b32 s2, ttmp7
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s4, s3, 26
	s_ashr_i32 s3, ttmp7, 31
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s8, s4
	v_mad_co_i64_i32 v[3:4], null, v3, s9, s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s2, s4, 6
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s3, s2, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[2:3], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 12
	v_lshlrev_b64_e32 v[3:4], 3, v[3:4]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v4, vcc_lo
	global_store_b64 v[0:1], v[6:7], off
.LBB1_18:                               ; %_Z22store_iu4_atiled_quant18iu4_quantized_lanePhiiiii.exit
	s_endpgm
.Lfunc_end1:
	.size	quantize_int4_mmq_ds128_atiled, .Lfunc_end1-quantize_int4_mmq_ds128_atiled
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel quantize_int4_mmq_ds128_atiled
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-quantize_int4_mmq_ds128_atiled)<<4)&4080)>>4
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
	.set .Lquantize_int4_mmq_ds128_atiled.num_vgpr, 36
	.set .Lquantize_int4_mmq_ds128_atiled.num_agpr, 0
	.set .Lquantize_int4_mmq_ds128_atiled.numbered_sgpr, 12
	.set .Lquantize_int4_mmq_ds128_atiled.num_named_barrier, 0
	.set .Lquantize_int4_mmq_ds128_atiled.private_seg_size, 0
	.set .Lquantize_int4_mmq_ds128_atiled.uses_vcc, 1
	.set .Lquantize_int4_mmq_ds128_atiled.uses_flat_scratch, 0
	.set .Lquantize_int4_mmq_ds128_atiled.has_dyn_sized_stack, 0
	.set .Lquantize_int4_mmq_ds128_atiled.has_recursion, 0
	.set .Lquantize_int4_mmq_ds128_atiled.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2668
; TotalNumSgprs: 14
; NumVgprs: 36
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 14
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
	.protected	gemm_mq4g256v2_residual_mmq_iu4 ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4
	.globl	gemm_mq4g256v2_residual_mmq_iu4
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4,@function
gemm_mq4g256v2_residual_mmq_iu4:        ; @gemm_mq4g256v2_residual_mmq_iu4
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[12:15], s[0:1], 0x18
	s_load_b128 s[16:19], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_lshl_b32 s22, ttmp9, 7
	s_lshl_b32 s24, ttmp7, 7
	v_lshrrev_b32_e32 v120, 5, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s22, s12
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s24, s14
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s11, s0, s1
	s_cmp_eq_u32 s15, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB2_15
; %bb.1:
	s_mov_b32 s28, 0
	s_and_b32 vcc_lo, exec_lo, s11
	s_cbranch_vccz .LBB2_16
; %bb.2:                                ; %.preheader559.i
	s_ashr_i32 s0, s13, 31
	s_add_co_i32 s2, s12, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s10, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_6
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s24, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB2_5
; %bb.4:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB2_5:                                ; %.preheader554.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s22, v1
	v_lshlrev_b32_e32 v2, 4, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[16:17]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v2, v2, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v2, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, 0, v1, v3
	ds_store_2addr_stride64_b32 v1, v4, v2 offset0:48 offset1:56
.LBB2_6:                                ; %Flow1703
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s24, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s22, v9
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v13, 16, v13
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s3, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s3, v3, v[1:2]
	global_load_b64 v[1:2], v68, s[18:19] offset:8
	global_load_b64 v[3:4], v69, s[16:17] offset:8
	global_load_b64 v[5:6], v70, s[18:19] offset:8
	global_load_b64 v[7:8], v71, s[16:17] offset:8
	v_bfe_u32 v12, v0, 1, 1
	v_and_or_b32 v9, v9, 15, v13
	v_or_b32_e32 v14, 8, v120
	v_cmp_gt_i32_e64 s0, s14, v10
	v_mov_b32_e32 v148, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_mov_b32_e32 v180, 0
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v160, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v143, 0
	v_bfe_u32 v185, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v156, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v177, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	ds_store_2addr_stride64_b64 v186, v[1:2], v[3:4] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[5:6], v[7:8] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_17
; %bb.7:                                ; %.preheader553.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v177, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v196, 0 :: v_dual_add_nc_u32 v5, s22, v1
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v7, 6, v185
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v160
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v10, 8, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v182, 0 :: v_dual_add_nc_u32 v11, s24, v1
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 2, v6
	v_dual_mov_b32 v178, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v4, 11, v2
	v_lshl_or_b32 v189, v3, 10, v2
	v_min_i32_e32 v190, s4, v11
	v_cmp_gt_i32_e64 s2, s14, v11
	v_add3_u32 v191, 0, v1, v12
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v192, 4, v6
	v_add3_u32 v193, 0, v10, v7
	v_add3_u32 v194, 0, v13, v8
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v195, 2, v6
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v121, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s29, 0x1000
	s_movk_i32 s23, 0x3000
	s_movk_i32 s25, 0x3800
	s_mov_b32 s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB2_9
.LBB2_8:                                ;   in Loop: Header=BB2_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_17
.LBB2_9:                                ; %.preheader553.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_11 Depth 2
	s_mov_b32 s7, s5
	s_add_co_i32 s26, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s26, s10
	s_mov_b32 s34, 0
	s_cselect_b32 s27, -1, 0
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s31, -1
	s_mov_b32 s33, 0
	s_branch .LBB2_11
.LBB2_10:                               ;   in Loop: Header=BB2_11 Depth=2
	s_wait_loadcnt_dscnt 0x30b
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v177, v177, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_mul_f32 v112, v100, v94 :: v_dual_fmac_f32 v59, v113, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v184, v184, v58
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_fmac_f32_e32 v57, v60, v95
	v_mul_f32_e32 v58, v112, v61
	v_mul_f32_e32 v60, v96, v94
	v_cvt_f32_i32_e32 v61, v63
	v_dual_add_f32 v182, v182, v59 :: v_dual_mul_f32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_add_f32 v183, v183, v57
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v63, v99, v94 :: v_dual_fmac_f32 v60, v61, v95
	s_wait_dscnt 0xa
	v_mul_f32_e32 v57, v110, v92
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v180, v180, v58 :: v_dual_add_f32 v181, v181, v59
	v_fmac_f32_e32 v62, v63, v95
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_mul_f32_e32 v57, v111, v92
	v_dual_add_f32 v178, v178, v60 :: v_dual_add_f32 v179, v179, v62
	v_mul_f32_e32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_add_f32_e32 v176, v176, v49
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v60, v59 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v52, v105, v92 :: v_dual_mul_f32 v57, v100, v92
	v_fmac_f32_e32 v51, v58, v59
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	v_dual_add_f32 v175, v175, v50 :: v_dual_mul_f32 v58, v102, v92
	v_fmac_f32_e32 v49, v52, v59
	v_dual_add_f32 v173, v173, v51 :: v_dual_mul_f32 v52, v96, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v55, v103, v92
	v_mul_f32_e32 v57, v98, v92
	v_cvt_f32_i32_e32 v56, v56
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	v_mul_f32_e32 v54, v57, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	v_dual_add_f32 v174, v174, v49 :: v_dual_add_f32 v171, v171, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v172, v172, v50
	s_wait_dscnt 0x9
	v_dual_fmac_f32 v54, v55, v59 :: v_dual_mul_f32 v49, v110, v90
	v_cvt_f32_i32_e32 v41, v41
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_mul_f32_e32 v42, v50, v42
	v_dual_mul_f32 v50, v106, v90 :: v_dual_add_f32 v169, v169, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v170, v170, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_dual_mul_f32 v52, v109, v90 :: v_dual_mul_f32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v42, v52, v51 :: v_dual_add_f32 v167, v167, v41
	v_mul_f32_e32 v41, v49, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v44, v105, v90 :: v_dual_fmac_f32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_dual_add_f32 v168, v168, v42 :: v_dual_add_f32 v165, v165, v43
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v41, v44, v51 :: v_dual_mul_f32 v42, v49, v45
	v_mul_f32_e32 v44, v96, v90
	v_cvt_f32_i32_e32 v45, v47
	v_mul_f32_e32 v47, v103, v90
	v_mul_f32_e32 v43, v50, v46
	v_dual_mul_f32 v46, v101, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v42, v46, v51 :: v_dual_fmac_f32 v43, v47, v51
	v_dual_mul_f32 v46, v49, v48 :: v_dual_mul_f32 v47, v99, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v44, v45, v51
	v_dual_add_f32 v166, v166, v41 :: v_dual_add_f32 v163, v163, v42
	s_wait_dscnt 0x8
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v41, v110, v72 :: v_dual_fmac_f32 v46, v47, v51
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_dual_add_f32 v164, v164, v43 :: v_dual_add_f32 v161, v161, v44
	v_add_f32_e32 v162, v162, v46
	v_cvt_f32_i32_e32 v43, v73
	v_mul_f32_e32 v33, v41, v33
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	v_dual_mul_f32 v35, v42, v35 :: v_dual_mul_f32 v42, v107, v72
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v44, v43
	v_dual_add_f32 v158, v158, v33 :: v_dual_mul_f32 v33, v41, v36
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v36, v105, v72
	v_fmac_f32_e32 v35, v42, v43
	v_dual_mul_f32 v42, v102, v72 :: v_dual_add_f32 v159, v159, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v34, v41, v37 :: v_dual_mul_f32 v37, v103, v72
	v_fmac_f32_e32 v33, v36, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v156, v156, v35 :: v_dual_mul_f32 v35, v42, v38
	v_dual_mul_f32 v36, v101, v72 :: v_dual_mul_f32 v41, v98, v72
	v_dual_mul_f32 v38, v96, v72 :: v_dual_add_f32 v157, v157, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	v_mul_f32_e32 v38, v99, v72
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_add_f32 v154, v154, v34 :: v_dual_fmac_f32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_mul_f32 v25, v39, v25
	v_dual_mul_f32 v26, v40, v26 :: v_dual_mul_f32 v37, v94, v87
	v_dual_mul_f32 v34, v94, v89 :: v_dual_add_f32 v155, v155, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v152, v152, v33 :: v_dual_add_f32 v153, v153, v36
	v_dual_fmac_f32 v26, v37, v95 :: v_dual_fmac_f32 v25, v34, v95
	s_wait_dscnt 0x4
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v150, v150, v25 :: v_dual_add_f32 v151, v151, v26
	v_mul_f32_e32 v26, v94, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v28, v29, v95
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v148, v148, v25 :: v_dual_mul_f32 v25, v26, v30
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_dual_add_f32 v147, v147, v28 :: v_dual_mul_f32 v28, v94, v75
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v25, v26, v95
	v_mul_f32_e32 v26, v29, v31
	v_add_f32_e32 v149, v149, v27
	v_mul_f32_e32 v29, v94, v77
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v20, v20
	v_fmac_f32_e32 v26, v28, v95
	v_dual_mul_f32 v31, v92, v86 :: v_dual_add_f32 v146, v146, v25
	v_mul_f32_e32 v27, v30, v32
	v_mul_f32_e32 v30, v92, v88
	v_mul_f32_e32 v28, v92, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v18, v31, v18
	v_cvt_f32_i32_e32 v19, v19
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v30, v92, v84
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v145, v145, v27 :: v_dual_mul_f32 v20, v30, v20
	v_dual_add_f32 v144, v144, v26 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v25, v92, v83 :: v_dual_mul_f32 v26, v92, v85
	v_mul_f32_e32 v19, v29, v19
	v_cvt_f32_i32_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v142, v142, v17
	v_fmac_f32_e32 v18, v28, v59
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_fmac_f32 v19, v25, v59
	v_mul_f32_e32 v17, v92, v80
	v_cvt_f32_i32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v143, v143, v18
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_dual_add_f32 v140, v140, v19 :: v_dual_add_f32 v141, v141, v20
	v_mul_f32_e32 v20, v92, v74
	v_cvt_f32_i32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_dual_mul_f32 v22, v92, v79 :: v_dual_mul_f32 v17, v17, v18
	v_mul_f32_e32 v18, v92, v81
	v_cvt_f32_i32_e32 v23, v24
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_cvt_f32_i32_e32 v11, v11
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v138, v138, v17
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	v_fmac_f32_e32 v20, v21, v59
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_mul_f32_e32 v21, v90, v88
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v139, v139, v19 :: v_dual_mul_f32 v10, v22, v10
	v_dual_add_f32 v136, v136, v20 :: v_dual_fmac_f32 v17, v18, v59
	v_mul_f32_e32 v18, v90, v89
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v9, v21, v9 :: v_dual_mul_f32 v20, v90, v82
	v_mul_f32_e32 v19, v90, v87
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v9, v18, v51
	v_dual_mul_f32 v18, v90, v85 :: v_dual_mul_f32 v21, v90, v84
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_mul_f32 v19, v90, v80
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v3, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v12, v21, v12
	v_add_f32_e32 v137, v137, v17
	v_mul_f32_e32 v17, v90, v83
	v_add_f32_e32 v135, v135, v10
	v_mul_f32_e32 v10, v90, v74
	v_fmac_f32_e32 v12, v18, v51
	v_dual_mul_f32 v11, v20, v11 :: v_dual_add_f32 v134, v134, v9
	v_dual_mul_f32 v20, v90, v78 :: v_dual_mul_f32 v9, v19, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_f32_e32 v133, v133, v12
	v_mul_f32_e32 v12, v90, v76
	v_cvt_f32_i32_e32 v4, v4
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v9, v14, v51
	v_add_f32_e32 v132, v132, v11
	v_cvt_f32_i32_e32 v11, v15
	v_cvt_f32_i32_e32 v14, v16
	s_wait_loadcnt 0x0
	v_add_f32_e32 v130, v130, v9
	s_barrier_signal -1
	v_mul_f32_e32 v9, v10, v11
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_xor_b32 s4, s31, -1
	s_mov_b32 s34, 1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v9, v10, v51
	v_dual_fmac_f32 v13, v17, v51 :: v_dual_mul_f32 v10, v72, v86
	v_mul_f32_e32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	v_add_f32_e32 v128, v128, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v131, v131, v13 :: v_dual_mul_f32 v2, v10, v2
	v_mul_f32_e32 v13, v90, v77
	v_mul_f32_e32 v9, v72, v87
	v_fmac_f32_e32 v1, v12, v43
	v_mul_f32_e32 v10, v72, v82
	v_mul_f32_e32 v12, v72, v84
	s_mov_b32 s31, 0
	v_fmac_f32_e32 v2, v9, v43
	v_dual_fmac_f32 v11, v13, v51 :: v_dual_add_f32 v126, v126, v1
	v_mul_f32_e32 v1, v10, v3
	v_mul_f32_e32 v3, v12, v4
	v_dual_mul_f32 v4, v72, v83 :: v_dual_mul_f32 v9, v72, v85
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v129, v129, v11 :: v_dual_mul_f32 v10, v72, v80
	v_mul_f32_e32 v11, v72, v78
	v_add_f32_e32 v127, v127, v2
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s33, -1
	v_mul_f32_e32 v2, v10, v5
	v_mul_f32_e32 v10, v72, v79
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_dual_mul_f32 v6, v72, v74 :: v_dual_mul_f32 v5, v72, v81
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_fmac_f32_e32 v4, v10, v43
	v_mul_f32_e32 v6, v6, v7
	v_dual_fmac_f32 v3, v9, v43 :: v_dual_add_f32 v124, v124, v1
	v_dual_mul_f32 v9, v72, v76 :: v_dual_fmac_f32 v2, v5, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v123, v123, v4
	v_add_f32_e32 v125, v125, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v7, v9, v8 :: v_dual_mul_f32 v8, v72, v75
	v_dual_mul_f32 v9, v72, v77 :: v_dual_add_f32 v122, v122, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v6, v8, v43 :: v_dual_fmac_f32 v7, v9, v43
	v_add_f32_e32 v121, v121, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_8
.LBB2_11:                               ;   Parent Loop BB2_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s34, s7
	s_lshl_b32 s36, s34, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[4:5], s[14:15]
	s_mov_b32 s37, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[38:39], 0x48
	s_add_nc_u64 s[36:37], s[8:9], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[18:19], s[38:39]
	v_add_nc_u32_e32 v80, s29, v189
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s35, s38, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s39, 0, s35
	v_add_co_u32 v3, s35, s36, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s37, 0, s35
	v_add_co_u32 v5, s35, s38, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s39, 0, s35
	v_add_co_u32 v7, s35, s36, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s37, 0, s35
	global_load_b64 v[118:119], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[116:117], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	v_add_nc_u32_e32 v81, s30, v188
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v81 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v81 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v80 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v81 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v81 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s0
	v_cndmask_b32_e64 v72, 0, v118, s0
	s_and_b32 s29, s33, s27
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s29
	s_wait_loadcnt 0x2
	ds_store_2addr_stride64_b64 v186, v[112:113], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s1
	v_cndmask_b32_e64 v72, 0, v116, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[114:115], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_13
; %bb.12:                               ;   in Loop: Header=BB2_11 Depth=2
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_add_co_i32 s4, s34, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_mul_u64 s[38:39], s[4:5], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v190, s[36:37]
	v_add_co_u32 v72, s30, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s37, 0, s30
	s_xor_b32 s30, s34, 1
	s_add_nc_u64 s[34:35], s[16:17], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s38, s30, 6
	s_mov_b32 s39, s5
	v_add_co_u32 v76, vcc_lo, v76, v195
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s4
	v_add_co_u32 v74, s40, s36, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s36, s34, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s37, 0, s40
	v_add_co_u32 v80, s34, s34, v71
	s_lshl_b32 s4, s30, 2
	v_add_co_ci_u32_e64 v79, null, s35, 0, s36
	v_add_co_ci_u32_e64 v81, null, s35, 0, s34
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[118:119], v[72:73], off offset:8
	global_load_b64 v[116:117], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB2_13:                               ; %.preheader551.i
                                        ;   in Loop: Header=BB2_11 Depth=2
	v_add_nc_u32_e32 v84, 0, v189
	v_add_nc_u32_e32 v85, 0, v188
	s_xor_b32 s4, s29, -1
	s_and_b32 s29, s31, exec_lo
	s_cselect_b32 s29, s23, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s30, s25, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v84
	v_add_nc_u32_e32 v80, 0x100, v85
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s30, v193
	v_add_nc_u32_e32 v72, s29, v194
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_movk_i32 s29, 0x2000
	s_movk_i32 s30, 0x4000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_10
; %bb.14:                               ; %.preheader552.i
                                        ;   in Loop: Header=BB2_11 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s4, s33, exec_lo
	s_cselect_b32 s4, s23, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s4, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s4, s25, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s4, v191
	s_mov_b32 s30, 0
	s_movk_i32 s29, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB2_10
.LBB2_15:
	s_mov_b32 s28, -1
.LBB2_16:                               ; %Flow1712
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s28
	s_cbranch_vccnz .LBB2_111
	s_branch .LBB2_219
.LBB2_17:                               ; %._crit_edge623.i
	v_mul_u32_u24_e32 v1, 0x500, v120
	v_lshlrev_b32_e32 v2, 2, v160
	s_add_co_i32 s0, s22, 0x80
	s_ashr_i32 s27, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v6, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s24, 0x80
	v_add3_u32 v8, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v4, 0x50, v160
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v6
	v_mad_u32_u24 v3, 0x280, v185, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_mov_b32 s26, s12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_108
; %bb.18:                               ; %.preheader548.i
	ds_store_2addr_b32 v3, v177, v184 offset1:20
	ds_store_2addr_b32 v3, v182, v183 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v180, v181 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v178, v179 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v1, s22, v160
	v_or_b32_e32 v9, s24, v6
	v_add3_u32 v7, 0, v4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v1
	v_cmp_gt_i32_e32 vcc_lo, s14, v9
	v_ashrrev_i32_e32 v2, 31, v1
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_mad_co_i64_i32 v[10:11], null, s26, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s0, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s0
	v_add_co_u32 v10, s0, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s0
	ds_load_b32 v13, v7
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off
.LBB2_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v10, 64, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v10
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_22
; %bb.21:
	v_mad_co_i64_i32 v[11:12], null, s26, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:1280
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off
.LBB2_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v11
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_24
; %bb.23:
	v_mad_co_i64_i32 v[11:12], null, s26, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:2560
	global_load_b32 v13, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:128
.LBB2_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_26
; %bb.25:
	v_mad_co_i64_i32 v[11:12], null, s26, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:3840
	global_load_b32 v13, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:128
.LBB2_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v11
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_28
; %bb.27:
	v_mad_co_i64_i32 v[11:12], null, s26, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:5120
	global_load_b32 v13, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:256
.LBB2_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_30
; %bb.29:
	v_mad_co_i64_i32 v[11:12], null, s26, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:6400
	global_load_b32 v13, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:256
.LBB2_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v11
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_32
; %bb.31:
	v_mad_co_i64_i32 v[11:12], null, s26, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v7 offset:7680
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384
.LBB2_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v11, 0x280, v185
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_34
; %bb.33:
	v_mad_co_i64_i32 v[12:13], null, s26, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s1, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s1
	v_add_co_u32 v12, s1, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v15, v7 offset:8960
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
.LBB2_34:                               ; %.preheader546.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v8, v8, v11
	v_or_b32_e32 v11, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v11
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v176, v175 offset1:20
	ds_store_2addr_b32 v8, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v172, v171 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v169, v170 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_36
; %bb.35:
	v_mad_co_i64_i32 v[12:13], null, s26, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s2, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s2
	v_add_co_u32 v12, s2, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v15, v7
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v12, 0x50, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v12
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_222
; %bb.37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_223
.LBB2_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_224
.LBB2_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_225
.LBB2_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_226
.LBB2_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_227
.LBB2_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_44
.LBB2_43:
	v_mad_co_i64_i32 v[13:14], null, s26, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:8960
	global_load_b32 v15, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:384
.LBB2_44:                               ; %.preheader546.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v13, 32, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v13
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v167, v168 offset1:20
	ds_store_2addr_b32 v8, v165, v166 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v163, v164 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v161, v162 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_46
; %bb.45:
	v_mad_co_i64_i32 v[14:15], null, s26, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s4, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s4
	v_add_co_u32 v14, s4, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s4
	ds_load_b32 v17, v7
	global_load_b32 v16, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off
.LBB2_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v14, 0x60, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v14
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_228
; %bb.47:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_229
.LBB2_48:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_230
.LBB2_49:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_231
.LBB2_50:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_232
.LBB2_51:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_233
.LBB2_52:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_54
.LBB2_53:
	v_mad_co_i64_i32 v[15:16], null, s26, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:8960
	global_load_b32 v17, v[15:16], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:384
.LBB2_54:                               ; %.preheader546.3.i
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v15, 48, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v15
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v158, v159 offset1:20
	ds_store_2addr_b32 v8, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s15, s6
	s_cbranch_execz .LBB2_56
; %bb.55:
	v_mad_co_i64_i32 v[16:17], null, s26, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s6, s20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s6
	v_add_co_u32 v16, s6, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s6
	ds_load_b32 v19, v7
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off
.LBB2_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v16, 0x70, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v16
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB2_234
; %bb.57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB2_235
.LBB2_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_236
.LBB2_59:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_237
.LBB2_60:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_238
.LBB2_61:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_239
.LBB2_62:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_64
.LBB2_63:
	v_mad_co_i64_i32 v[17:18], null, s26, v16, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:8960
	global_load_b32 v19, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:384
.LBB2_64:                               ; %.preheader547.1.i
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v17, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v17
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v150, v151 offset1:20
	ds_store_2addr_b32 v8, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v147, v146 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_66
; %bb.65:
	v_mad_co_i64_i32 v[17:18], null, s26, v9, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s8, s20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s8
	v_add_co_u32 v17, s8, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s8
	ds_load_b32 v20, v7
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
.LBB2_66:
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_68
; %bb.67:
	v_mad_co_i64_i32 v[17:18], null, s26, v10, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s8, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s8
	v_add_co_u32 v17, s8, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s8
	ds_load_b32 v20, v7 offset:1280
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
.LBB2_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v17, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v17
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_70
; %bb.69:
	v_mad_co_i64_i32 v[17:18], null, s26, v9, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s9, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s9
	v_add_co_u32 v17, s9, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s9
	ds_load_b32 v20, v7 offset:2560
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
.LBB2_70:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_72
; %bb.71:
	v_mad_co_i64_i32 v[17:18], null, s26, v10, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s9, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s9
	v_add_co_u32 v17, s9, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s9
	ds_load_b32 v20, v7 offset:3840
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
.LBB2_72:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v17, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v17
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB2_74
; %bb.73:
	v_mad_co_i64_i32 v[17:18], null, s26, v9, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s10, s20, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s10
	v_add_co_u32 v17, s10, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s10
	ds_load_b32 v20, v7 offset:5120
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
.LBB2_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB2_76
; %bb.75:
	v_mad_co_i64_i32 v[17:18], null, s26, v10, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s10, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s10
	v_add_co_u32 v17, s10, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s10
	ds_load_b32 v20, v7 offset:6400
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
.LBB2_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v17, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v17
	s_and_b32 s23, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s23
	s_cbranch_execz .LBB2_78
; %bb.77:
	v_mad_co_i64_i32 v[17:18], null, s26, v9, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v9, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v9, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v19, v7 offset:7680
	global_load_b32 v9, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v19, v9
	global_store_b32 v[17:18], v9, off offset:448
.LBB2_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s15, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_80
; %bb.79:
	v_mad_co_i64_i32 v[9:10], null, s26, v10, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:8960
	global_load_b32 v17, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:448
.LBB2_80:                               ; %.preheader546.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s15, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v142, v143 offset1:20
	ds_store_2addr_b32 v8, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_240
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_241
.LBB2_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_242
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_243
.LBB2_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_244
.LBB2_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB2_245
.LBB2_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_246
.LBB2_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_89
.LBB2_88:
	v_mad_co_i64_i32 v[9:10], null, s26, v12, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:8960
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
.LBB2_89:                               ; %.preheader546.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v134, v135 offset1:20
	ds_store_2addr_b32 v8, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_247
; %bb.90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_248
.LBB2_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_249
.LBB2_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_250
.LBB2_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_251
.LBB2_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_252
.LBB2_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_253
.LBB2_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_98
.LBB2_97:
	v_mad_co_i64_i32 v[9:10], null, s26, v14, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:8960
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
.LBB2_98:                               ; %.preheader546.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v126, v127 offset1:20
	ds_store_2addr_b32 v8, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_254
; %bb.99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_255
.LBB2_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_256
.LBB2_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_257
.LBB2_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_258
.LBB2_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_259
.LBB2_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_260
.LBB2_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_107
.LBB2_106:
	v_mad_co_i64_i32 v[8:9], null, s26, v16, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	ds_load_b32 v7, v7 offset:8960
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v8, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v9, v2, vcc_lo
	global_load_b32 v8, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	global_store_b32 v[1:2], v7, off offset:448
.LBB2_107:                              ; %.loopexit.loopexit656.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB2_108:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_110
; %bb.109:                              ; %.preheader545.i
	ds_store_2addr_b32 v3, v177, v184 offset1:20
	ds_store_2addr_b32 v3, v182, v183 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v180, v181 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v178, v179 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_lo_u32 v1, s12, v6
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s23, s22, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[26:27], s[24:25]
	s_lshl_b64 s[2:3], s[22:23], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v4, 0, v4, v5
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v6, v1, v160, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, -1
	s_lshl_b32 s5, s12, 8
	s_movk_i32 s4, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v7, v6, s[0:3], null offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_or_b32 s6, s5, 0x80
	s_mul_i32 s7, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s7, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v7
	buffer_store_b32 v1, v6, s[0:3], null offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s6 offen
	s_movk_i32 s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s6, s5, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s6 offen
	s_movk_i32 s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_add_co_i32 s6, s5, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s6 offen
	s_lshl_b32 s4, s12, 6
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v176, v175 offset1:20
	ds_store_2addr_b32 v3, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v172, v171 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v169, v170 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s7 offen
	buffer_load_b32 v5, v6, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v1, v6, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s8 offen
	buffer_load_b32 v5, v6, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s8, s7, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v1, v6, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s8 offen
	buffer_load_b32 v5, v6, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_add_co_i32 s8, s7, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v1, v6, s[0:3], s8 offen
	s_lshl_b32 s6, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s6, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s8 offen
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s8, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v167, v168 offset1:20
	ds_store_2addr_b32 v3, v165, v166 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v163, v164 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v161, v162 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v1, v6, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s8 offen
	buffer_load_b32 v5, v6, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s9 offen
	buffer_load_b32 v1, v6, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s10 offen
	buffer_load_b32 v5, v6, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s10, s8, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s9 offen
	buffer_load_b32 v1, v6, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s10 offen
	buffer_load_b32 v5, v6, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_add_co_i32 s10, s8, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s9 offen
	buffer_load_b32 v1, v6, s[0:3], s10 offen
	s_mul_i32 s9, s12, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s15, s9, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s10 offen
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s23, s10, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v158, v159 offset1:20
	ds_store_2addr_b32 v3, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s9 offen
	buffer_load_b32 v1, v6, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s10 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_add_co_i32 s15, s9, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s23, s10, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_add_co_i32 s15, s9, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_add_co_i32 s23, s10, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_mov_b32 s15, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	s_or_b32 s23, s5, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v150, v151 offset1:20
	ds_store_2addr_b32 v3, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v147, v146 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_movk_i32 s15, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_or_b32 s23, s5, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_movk_i32 s15, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s23, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s23 offen
	s_movk_i32 s15, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s23 offen
	buffer_load_b32 v5, v6, s[0:3], s15 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s15, s7, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s5, s4, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v142, v143 offset1:20
	ds_store_2addr_b32 v3, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v1, v6, s[0:3], s15 offen
	s_add_co_i32 s5, s4, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v5, v6, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_add_co_i32 s15, s7, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v1, v6, s[0:3], s15 offen
	s_add_co_i32 s5, s4, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v5, v6, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s15, s7, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v1, v6, s[0:3], s15 offen
	s_or_b32 s5, s8, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s15 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s7 offen
	s_or_b32 s4, s6, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s7 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v134, v135 offset1:20
	ds_store_2addr_b32 v3, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_add_co_i32 s5, s8, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0x140
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v5, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s5, s8, 0x140
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v5, v6, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_add_co_i32 s5, s10, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v5
	buffer_store_b32 v1, v6, s[0:3], s6 offen
	buffer_load_b32 v1, v6, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s8 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v126, v127 offset1:20
	ds_store_2addr_b32 v3, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v3, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v3, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:10 offset1:15
	s_add_co_i32 s5, s10, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v3, v6, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:20 offset1:25
	s_add_co_i32 s5, s10, 0x140
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	buffer_store_b32 v1, v6, s[0:3], s4 offen
	buffer_load_b32 v1, v6, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s5 offen
	buffer_load_b32 v3, v6, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[1:2], v4 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v3
	buffer_store_b32 v1, v6, s[0:3], s9 offen
	buffer_load_b32 v1, v6, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v6, s[0:3], s10 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB2_110:                              ; %Flow1698
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_219
.LBB2_111:
	s_and_b32 vcc_lo, exec_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_219
; %bb.112:                              ; %.preheader552.i17
	s_ashr_i32 s0, s13, 31
	s_add_co_i32 s2, s12, -1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s0, s0, 24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s13, s0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s10, s0, 8
	s_mov_b32 s0, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_116
; %bb.113:                              ; %.lr.ph.i121
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s24, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB2_115
; %bb.114:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB2_115:                              ; %.preheader547.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s22, v1
	v_lshlrev_b32_e32 v2, 4, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[16:17]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v2, v2, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v2, v2.l
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v2, 0, v2 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, 0, v1, v3
	ds_store_2addr_stride64_b32 v1, v4, v2 offset0:48 offset1:56
.LBB2_116:                              ; %Flow1711
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v156, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s24, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s22, v9
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v13, 16, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s3, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s3, v3, v[1:2]
	global_load_b64 v[1:2], v68, s[18:19] offset:8
	global_load_b64 v[3:4], v69, s[16:17] offset:8
	global_load_b64 v[5:6], v70, s[18:19] offset:8
	global_load_b64 v[7:8], v71, s[16:17] offset:8
	v_bfe_u32 v12, v0, 1, 1
	v_and_or_b32 v9, v9, 15, v13
	v_or_b32_e32 v14, 8, v120
	v_cmp_gt_i32_e64 s0, s14, v10
	v_mov_b32_e32 v148, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_mov_b32_e32 v182, 0
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v144, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v143, 0
	v_bfe_u32 v161, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	ds_store_2addr_stride64_b64 v186, v[1:2], v[3:4] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[5:6], v[7:8] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_125
; %bb.117:                              ; %.preheader546.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v5, s22, v1
	v_dual_mov_b32 v185, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v7, 6, v161
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v144
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v10, 8, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v184, 0 :: v_dual_add_nc_u32 v11, s24, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 2, v6
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v4, 11, v2
	v_lshl_or_b32 v189, v3, 10, v2
	v_min_i32_e32 v190, s4, v11
	v_cmp_gt_i32_e64 s2, s14, v11
	v_add3_u32 v191, 0, v1, v12
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v192, 4, v6
	v_add3_u32 v193, 0, v10, v7
	v_add3_u32 v194, 0, v13, v8
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v195, 2, v6
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v121, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s11, 0x3000
	s_movk_i32 s13, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB2_119
.LBB2_118:                              ;   in Loop: Header=BB2_119 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s6, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_125
.LBB2_119:                              ; %.preheader546.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_121 Depth 2
	s_mov_b32 s7, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s23, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, s10
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s28, -1
	s_mov_b32 s29, 0
	s_branch .LBB2_121
.LBB2_120:                              ;   in Loop: Header=BB2_121 Depth=2
	s_wait_loadcnt_dscnt 0x30b
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_dual_add_f32 v178, v178, v57 :: v_dual_mul_f32 v57, v112, v60
	v_mul_f32_e32 v60, v105, v94
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v185, v185, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v183, v183, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v184, v184, v57 :: v_dual_add_f32 v181, v181, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_dual_add_f32 v182, v182, v59 :: v_dual_add_f32 v179, v179, v60
	v_add_f32_e32 v180, v180, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v177, v177, v49
	v_dual_mul_f32 v49, v57, v52 :: v_dual_fmac_f32 v50, v60, v59
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v52, v105, v92 :: v_dual_fmac_f32 v51, v58, v59
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v58, v102, v92
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	v_add_f32_e32 v176, v176, v50
	v_fmac_f32_e32 v49, v52, v59
	v_add_f32_e32 v174, v174, v51
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_mul_f32_e32 v52, v96, v92
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v57, v98, v92
	v_mul_f32_e32 v55, v103, v92
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v54, v57, v56 :: v_dual_add_f32 v175, v175, v49
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	s_wait_dscnt 0x9
	v_mul_f32_e32 v49, v110, v90
	v_cvt_f32_i32_e32 v41, v41
	v_dual_add_f32 v173, v173, v50 :: v_dual_add_f32 v172, v172, v51
	v_fmac_f32_e32 v54, v55, v59
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_dual_mul_f32 v49, v111, v90 :: v_dual_add_f32 v170, v170, v52
	v_add_f32_e32 v171, v171, v54
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v168, v168, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_dual_fmac_f32 v43, v50, v51 :: v_dual_mul_f32 v44, v105, v90
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v169, v169, v42
	v_dual_fmac_f32 v41, v44, v51 :: v_dual_mul_f32 v42, v49, v45
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v45, v47
	v_dual_add_f32 v166, v166, v43 :: v_dual_mul_f32 v47, v103, v90
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_cvt_f32_i32_e32 v48, v48
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v167, v167, v41 :: v_dual_fmac_f32 v44, v45, v51
	s_wait_dscnt 0x8
	v_mul_f32_e32 v41, v110, v72
	v_cvt_f32_i32_e32 v33, v33
	v_dual_add_f32 v164, v164, v42 :: v_dual_add_f32 v165, v165, v43
	v_fmac_f32_e32 v46, v47, v51
	v_mul_f32_e32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v43, v73
	v_mul_f32_e32 v33, v41, v33
	v_mul_f32_e32 v41, v111, v72
	v_dual_add_f32 v162, v162, v44 :: v_dual_add_f32 v163, v163, v46
	v_mul_f32_e32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	v_cvt_f32_i32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v35, v42, v35 :: v_dual_mul_f32 v42, v107, v72
	v_dual_add_f32 v159, v159, v33 :: v_dual_fmac_f32 v34, v44, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v33, v41, v36
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v36, v105, v72
	v_fmac_f32_e32 v35, v42, v43
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	v_add_f32_e32 v160, v160, v34
	v_dual_mul_f32 v42, v102, v72 :: v_dual_fmac_f32 v33, v36, v43
	v_add_f32_e32 v157, v157, v35
	v_mul_f32_e32 v34, v41, v37
	v_dual_mul_f32 v36, v101, v72 :: v_dual_mul_f32 v37, v103, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v38, v96, v72
	v_dual_mul_f32 v41, v98, v72 :: v_dual_fmac_f32 v34, v36, v43
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v158, v158, v33 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_mul_f32_e32 v38, v99, v72
	v_add_f32_e32 v155, v155, v34
	v_fmac_f32_e32 v33, v37, v43
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v156, v156, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v153, v153, v33
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v151, v151, v25 :: v_dual_add_f32 v152, v152, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	v_fmac_f32_e32 v28, v29, v95
	v_add_f32_e32 v149, v149, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v25, v26, v30
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v150, v150, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_f32 v148, v148, v28 :: v_dual_fmac_f32 v25, v26, v95
	v_mul_f32_e32 v28, v94, v75
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v147, v147, v25 :: v_dual_fmac_f32 v26, v28, v95
	v_fmac_f32_e32 v27, v29, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v30, v92, v84
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v28, v92, v87
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v145, v145, v26 :: v_dual_fmac_f32 v18, v28, v59
	v_mul_f32_e32 v20, v30, v20
	v_dual_mul_f32 v26, v92, v85 :: v_dual_fmac_f32 v17, v25, v59
	v_mul_f32_e32 v25, v92, v83
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v146, v146, v27 :: v_dual_add_f32 v143, v143, v18
	v_fmac_f32_e32 v20, v26, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v19, v29, v19 :: v_dual_add_f32 v142, v142, v17
	v_mul_f32_e32 v17, v92, v80
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_dual_add_f32 v141, v141, v20 :: v_dual_mul_f32 v20, v92, v74
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_dual_add_f32 v140, v140, v19 :: v_dual_mul_f32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v13, v13
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v5, v5
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_fmac_f32 v19, v22, v59
	v_dual_mul_f32 v21, v90, v88 :: v_dual_mul_f32 v22, v90, v86
	v_add_f32_e32 v138, v138, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v136, v136, v20 :: v_dual_add_f32 v139, v139, v19
	v_dual_mul_f32 v9, v21, v9 :: v_dual_mul_f32 v20, v90, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v10, v22, v10
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_mul_f32_e32 v19, v90, v87
	v_dual_mul_f32 v11, v20, v11 :: v_dual_mul_f32 v12, v21, v12
	v_mul_f32_e32 v20, v90, v78
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v17, v18, v59
	v_mul_f32_e32 v18, v90, v89
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_mul_f32 v19, v90, v80
	v_cvt_f32_i32_e32 v6, v6
	v_add_f32_e32 v137, v137, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v9, v18, v51
	v_dual_mul_f32 v17, v90, v83 :: v_dual_mul_f32 v18, v90, v85
	v_add_f32_e32 v135, v135, v10
	v_mul_f32_e32 v10, v90, v74
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_fmac_f32_e32 v12, v18, v51
	v_add_f32_e32 v134, v134, v9
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	v_add_f32_e32 v133, v133, v12
	v_mul_f32_e32 v12, v90, v76
	v_fmac_f32_e32 v9, v14, v51
	v_add_f32_e32 v132, v132, v11
	v_cvt_f32_i32_e32 v11, v15
	v_cvt_f32_i32_e32 v14, v16
	v_cvt_f32_i32_e32 v7, v7
	v_add_f32_e32 v130, v130, v9
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v10, v90, v75
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_add_f32_e32 v154, v154, v36
	s_xor_b32 s4, s28, -1
	v_dual_fmac_f32 v9, v10, v51 :: v_dual_mul_f32 v10, v72, v86
	v_fmac_f32_e32 v13, v17, v51
	s_mov_b32 s30, 1
	s_mov_b32 s28, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v128, v128, v9
	v_dual_mul_f32 v2, v10, v2 :: v_dual_add_f32 v131, v131, v13
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v11, v12, v14
	v_dual_mul_f32 v12, v72, v88 :: v_dual_mul_f32 v13, v90, v77
	v_mul_f32_e32 v10, v72, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v2, v9, v43 :: v_dual_mul_f32 v9, v72, v85
	v_mul_f32_e32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	v_add_f32_e32 v127, v127, v2
	s_mov_b32 s29, -1
	s_barrier_wait -1
	v_fmac_f32_e32 v1, v12, v43
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v11, v13, v51
	global_inv scope:SCOPE_SE
	v_add_f32_e32 v126, v126, v1
	v_mul_f32_e32 v1, v10, v3
	v_add_f32_e32 v129, v129, v11
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	v_dual_mul_f32 v10, v72, v80 :: v_dual_mul_f32 v11, v72, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v2, v10, v5
	v_mul_f32_e32 v10, v72, v79
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_dual_mul_f32 v6, v72, v74 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_mul_f32_e32 v6, v6, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v3, v9, v43 :: v_dual_add_f32 v124, v124, v1
	v_dual_mul_f32 v9, v72, v76 :: v_dual_fmac_f32 v2, v5, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v123, v123, v4
	v_add_f32_e32 v125, v125, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v7, v9, v8 :: v_dual_mul_f32 v8, v72, v75
	v_dual_mul_f32 v9, v72, v77 :: v_dual_add_f32 v122, v122, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v6, v8, v43 :: v_dual_fmac_f32 v7, v9, v43
	v_add_f32_e32 v121, v121, v6
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_118
.LBB2_121:                              ;   Parent Loop BB2_119 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s30, s7
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_mov_b32 s35, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[8:9], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	v_add_nc_u32_e32 v80, s26, v189
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	v_add_co_u32 v3, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	v_add_co_u32 v5, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v7, s31, s34, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s31
	global_load_b64 v[118:119], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[116:117], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	v_add_nc_u32_e32 v81, s27, v188
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v81 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v81 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v80 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v81 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v81 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s0
	v_cndmask_b32_e64 v72, 0, v118, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_wait_loadcnt 0x2
	ds_store_2addr_stride64_b64 v186, v[112:113], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s1
	v_cndmask_b32_e64 v72, 0, v116, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[114:115], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_123
; %bb.122:                              ;   in Loop: Header=BB2_121 Depth=2
	s_add_co_i32 s4, s4, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_add_co_i32 s4, s30, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[4:5], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v190, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[16:17], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s5
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s4
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s4, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[118:119], v[72:73], off offset:8
	global_load_b64 v[116:117], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB2_123:                              ; %.preheader544.i
                                        ;   in Loop: Header=BB2_121 Depth=2
	v_add_nc_u32_e32 v84, 0, v189
	v_add_nc_u32_e32 v85, 0, v188
	s_xor_b32 s4, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s11, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s13, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v84
	v_add_nc_u32_e32 v80, 0x100, v85
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s27, v193
	v_add_nc_u32_e32 v72, s26, v194
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_120
; %bb.124:                              ; %.preheader545.i115
                                        ;   in Loop: Header=BB2_121 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s4, s29, exec_lo
	s_cselect_b32 s4, s11, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s4, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s4, s13, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s4, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB2_120
.LBB2_125:                              ; %._crit_edge616.i
	s_add_co_i32 s0, s22, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_mul_u32_u24_e32 v1, 0x500, v120
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s24, 0x80
	v_lshlrev_b32_e32 v3, 2, v144
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_lshrrev_b32_e32 v2, 4, v0
	s_cselect_b32 s1, -1, 0
	v_add3_u32 v3, 0, v1, v3
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_216
; %bb.126:                              ; %.preheader541.i
	v_mad_u32_u24 v0, 0x280, v161, v3
	v_or_b32_e32 v6, s24, v2
	v_mul_u32_u24_e32 v1, 0x50, v144
	v_lshlrev_b32_e32 v4, 2, v2
	ds_store_2addr_b32 v0, v178, v185 offset1:20
	ds_store_2addr_b32 v0, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v0, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v0, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s22, v144
	v_cmp_gt_i32_e32 vcc_lo, s14, v6
	v_add3_u32 v4, 0, v1, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_128
; %bb.127:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	ds_load_b32 v5, v4
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s0, s20, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s0, v7, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s0
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v5, off
.LBB2_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, 64, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v7
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_130
; %bb.129:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v5, v4 offset:1280
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off
.LBB2_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v5
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_132
; %bb.131:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v5, v4 offset:2560
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off offset:128
.LBB2_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_134
; %bb.133:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v5, v4 offset:3840
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off offset:128
.LBB2_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v5
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_136
; %bb.135:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v5, v4 offset:5120
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off offset:256
.LBB2_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_138
; %bb.137:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v5, v4 offset:6400
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off offset:256
.LBB2_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v5
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_140
; %bb.139:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v5, v4 offset:7680
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v5, off offset:384
.LBB2_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v5, 0x280, v161
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_142
; %bb.141:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v4 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:384
.LBB2_142:                              ; %.preheader539.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v3, v5
	v_or_b32_e32 v8, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v8
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v177, v176 offset1:20
	ds_store_2addr_b32 v5, v174, v175 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v173, v172 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v170, v171 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_144
; %bb.143:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v4
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s2, s20, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s2, v9, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB2_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v9, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v9
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_261
; %bb.145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_262
.LBB2_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_263
.LBB2_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_264
.LBB2_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_265
.LBB2_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_266
.LBB2_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_152
.LBB2_151:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v4 offset:8960
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB2_152:                              ; %.preheader539.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v10, 32, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v10
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v168, v169 offset1:20
	ds_store_2addr_b32 v5, v166, v167 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v164, v165 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v162, v163 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_154
; %bb.153:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v4
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s4, s20, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s4, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s4
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB2_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v11, 0x60, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v11
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_267
; %bb.155:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_268
.LBB2_156:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_269
.LBB2_157:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_270
.LBB2_158:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_271
.LBB2_159:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_272
.LBB2_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_162
.LBB2_161:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v4 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
.LBB2_162:                              ; %.preheader539.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 48, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v12
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v159, v160 offset1:20
	ds_store_2addr_b32 v5, v157, v158 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v155, v156 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v153, v154 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB2_164
; %bb.163:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	ds_load_b32 v17, v4
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s6, s20, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s6, v13, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s6
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v17, off
.LBB2_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v13, 0x70, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v13
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_273
; %bb.165:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_274
.LBB2_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_275
.LBB2_167:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_276
.LBB2_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_277
.LBB2_169:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_278
.LBB2_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_172
.LBB2_171:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v4 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
.LBB2_172:                              ; %.preheader540.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v14
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v151, v152 offset1:20
	ds_store_2addr_b32 v5, v149, v150 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v148, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v145, v146 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_174
; %bb.173:
	v_mad_co_i64_i32 v[14:15], null, s12, v6, 0
	ds_load_b32 v18, v4
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s8, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s8, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s8
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:64
.LBB2_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_176
; %bb.175:
	v_mad_co_i64_i32 v[14:15], null, s12, v7, 0
	ds_load_b32 v18, v4 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s8, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s8, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s8
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:64
.LBB2_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v14, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v14
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_178
; %bb.177:
	v_mad_co_i64_i32 v[14:15], null, s12, v6, 0
	ds_load_b32 v18, v4 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s9, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s9, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s9
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:192
.LBB2_178:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_180
; %bb.179:
	v_mad_co_i64_i32 v[14:15], null, s12, v7, 0
	ds_load_b32 v18, v4 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s9, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s9, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s9
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:192
.LBB2_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v14, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v14
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_182
; %bb.181:
	v_mad_co_i64_i32 v[14:15], null, s12, v6, 0
	ds_load_b32 v18, v4 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s10, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s10, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s10
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:320
.LBB2_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_184
; %bb.183:
	v_mad_co_i64_i32 v[14:15], null, s12, v7, 0
	ds_load_b32 v18, v4 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s10, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s10, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s10
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:320
.LBB2_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v14, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v14
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB2_186
; %bb.185:
	v_mad_co_i64_i32 v[14:15], null, s12, v6, 0
	ds_load_b32 v6, v4 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, vcc_lo, s20, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, vcc_lo, v14, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v6, off offset:448
.LBB2_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_188
; %bb.187:
	v_mad_co_i64_i32 v[6:7], null, s12, v7, 0
	ds_load_b32 v16, v4 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:448
.LBB2_188:                              ; %.preheader539.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v142, v143 offset1:20
	ds_store_2addr_b32 v5, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_279
; %bb.189:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_280
.LBB2_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_281
.LBB2_191:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_282
.LBB2_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_283
.LBB2_193:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_284
.LBB2_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_285
.LBB2_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_197
.LBB2_196:
	v_mad_co_i64_i32 v[6:7], null, s12, v9, 0
	ds_load_b32 v14, v4 offset:8960
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:448
.LBB2_197:                              ; %.preheader539.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v134, v135 offset1:20
	ds_store_2addr_b32 v5, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_286
; %bb.198:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_287
.LBB2_199:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_288
.LBB2_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_289
.LBB2_201:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_290
.LBB2_202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_291
.LBB2_203:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_292
.LBB2_204:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_206
.LBB2_205:
	v_mad_co_i64_i32 v[6:7], null, s12, v11, 0
	ds_load_b32 v10, v4 offset:8960
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:448
.LBB2_206:                              ; %.preheader539.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v126, v127 offset1:20
	ds_store_2addr_b32 v5, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_293
; %bb.207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_294
.LBB2_208:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_295
.LBB2_209:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_296
.LBB2_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_297
.LBB2_211:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_298
.LBB2_212:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_299
.LBB2_213:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_215
.LBB2_214:
	v_mad_co_i64_i32 v[5:6], null, s12, v13, 0
	ds_load_b32 v7, v4 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v4, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v7, off offset:448
.LBB2_215:                              ; %.loopexit.loopexit649.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB2_216:                              ; %Flow1705
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_218
; %bb.217:                              ; %.preheader538.i
	v_mad_u32_u24 v9, 0x280, v161, v3
	v_mul_u32_u24_e32 v0, 0x50, v144
	v_lshlrev_b32_e32 v1, 2, v2
	v_mul_lo_u32 v2, s12, v2
	s_ashr_i32 s25, s24, 31
	ds_store_2addr_b32 v9, v178, v185 offset1:20
	ds_store_2addr_b32 v9, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add3_u32 v10, 0, v0, v1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[24:25]
	s_ashr_i32 s23, s22, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[4:5], s[22:23], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v11, v2, v144, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_mov_b32 s3, 0x31004000
	s_mov_b32 s2, -1
	s_lshl_b32 s6, s12, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_movk_i32 s7, 0x80
	s_movk_i32 s8, 0x100
	s_movk_i32 s9, 0x180
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_or_b32 s10, s6, 0x80
	s_add_co_i32 s4, s6, 0x100
	s_add_co_i32 s5, s6, 0x180
	s_add_co_i32 s17, s6, 0x140
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], null offen
	buffer_store_b32 v1, v11, s[0:3], s6 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	buffer_store_b32 v4, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s8 offen
	buffer_store_b32 v6, v11, s[0:3], s4 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s9 offen
	buffer_store_b32 v8, v11, s[0:3], s5 offen
	s_lshl_b32 s4, s12, 6
	s_mul_i32 s5, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 0x80
	s_add_co_i32 s8, s5, 0x80
	s_add_co_i32 s9, s4, 0x100
	s_add_co_i32 s10, s5, 0x100
	s_add_co_i32 s11, s4, 0x180
	s_add_co_i32 s13, s5, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v177, v176 offset1:20
	ds_store_2addr_b32 v9, v174, v175 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v173, v172 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v170, v171 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s7 offen
	buffer_store_b32 v3, v11, s[0:3], s8 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s9 offen
	buffer_store_b32 v5, v11, s[0:3], s10 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s11 offen
	buffer_store_b32 v7, v11, s[0:3], s13 offen
	s_lshl_b32 s7, s12, 7
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s7, 0x80
	s_add_co_i32 s10, s8, 0x80
	s_add_co_i32 s11, s7, 0x100
	s_add_co_i32 s13, s8, 0x100
	s_add_co_i32 s14, s7, 0x180
	s_add_co_i32 s15, s8, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v168, v169 offset1:20
	ds_store_2addr_b32 v9, v166, v167 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v164, v165 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v162, v163 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s7 offen
	buffer_store_b32 v1, v11, s[0:3], s8 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s9 offen
	buffer_store_b32 v3, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s11 offen
	buffer_store_b32 v5, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s14 offen
	buffer_store_b32 v7, v11, s[0:3], s15 offen
	s_mul_i32 s9, s12, 0xc0
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_add_co_i32 s12, s10, 0x80
	s_add_co_i32 s13, s9, 0x100
	s_add_co_i32 s14, s10, 0x100
	s_add_co_i32 s15, s9, 0x180
	s_add_co_i32 s16, s10, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v159, v160 offset1:20
	ds_store_2addr_b32 v9, v157, v158 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v155, v156 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v153, v154 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s9 offen
	buffer_store_b32 v1, v11, s[0:3], s10 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s11 offen
	buffer_store_b32 v3, v11, s[0:3], s12 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s13 offen
	buffer_store_b32 v5, v11, s[0:3], s14 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s15 offen
	buffer_store_b32 v7, v11, s[0:3], s16 offen
	s_mov_b32 s14, 64
	s_or_b32 s15, s6, 64
	s_movk_i32 s11, 0x140
	s_movk_i32 s12, 0xc0
	s_movk_i32 s13, 0x1c0
	s_or_b32 s16, s6, 0xc0
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v151, v152 offset1:20
	ds_store_2addr_b32 v9, v149, v150 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v148, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v145, v146 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s14 offen
	buffer_store_b32 v1, v11, s[0:3], s15 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s12 offen
	buffer_store_b32 v3, v11, s[0:3], s16 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s11 offen
	buffer_store_b32 v5, v11, s[0:3], s17 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s13 offen
	buffer_store_b32 v7, v11, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 64
	s_add_co_i32 s11, s5, 64
	s_add_co_i32 s12, s4, 0xc0
	s_add_co_i32 s13, s5, 0xc0
	s_add_co_i32 s14, s4, 0x140
	s_add_co_i32 s15, s5, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v142, v143 offset1:20
	ds_store_2addr_b32 v9, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s6 offen
	buffer_store_b32 v1, v11, s[0:3], s11 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s12 offen
	buffer_store_b32 v3, v11, s[0:3], s13 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s14 offen
	buffer_store_b32 v5, v11, s[0:3], s15 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s4 offen
	buffer_store_b32 v7, v11, s[0:3], s5 offen
	s_or_b32 s4, s7, 64
	s_or_b32 s5, s8, 64
	s_add_co_i32 s6, s7, 0xc0
	s_add_co_i32 s11, s8, 0xc0
	s_add_co_i32 s12, s7, 0x140
	s_add_co_i32 s13, s8, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v134, v135 offset1:20
	ds_store_2addr_b32 v9, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s6 offen
	buffer_store_b32 v3, v11, s[0:3], s11 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s12 offen
	buffer_store_b32 v5, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s7 offen
	buffer_store_b32 v7, v11, s[0:3], s8 offen
	s_add_co_i32 s4, s9, 64
	s_add_co_i32 s5, s10, 64
	s_add_co_i32 s6, s9, 0xc0
	s_add_co_i32 s7, s10, 0xc0
	s_add_co_i32 s8, s9, 0x140
	s_add_co_i32 s11, s10, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v9, v126, v127 offset1:20
	ds_store_2addr_b32 v9, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s6 offen
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s8 offen
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s9 offen
	buffer_store_b32 v7, v11, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB2_218:                              ; %Flow1706
	s_mov_b32 s0, -1
.LBB2_219:                              ; %Flow1714
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_221
; %bb.220:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1ELb0ELb0EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB2_221:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1ELb0ELb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB2_222:
	v_mad_co_i64_i32 v[13:14], null, s26, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:1280
	global_load_b32 v15, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_38
.LBB2_223:
	v_mad_co_i64_i32 v[13:14], null, s26, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:2560
	global_load_b32 v15, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_39
.LBB2_224:
	v_mad_co_i64_i32 v[13:14], null, s26, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:3840
	global_load_b32 v15, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_40
.LBB2_225:
	v_mad_co_i64_i32 v[13:14], null, s26, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:5120
	global_load_b32 v15, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_41
.LBB2_226:
	v_mad_co_i64_i32 v[13:14], null, s26, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:6400
	global_load_b32 v15, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_42
.LBB2_227:
	v_mad_co_i64_i32 v[13:14], null, s26, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s3, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s3
	v_add_co_u32 v13, s3, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s3
	ds_load_b32 v16, v7 offset:7680
	global_load_b32 v15, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_43
	s_branch .LBB2_44
.LBB2_228:
	v_mad_co_i64_i32 v[15:16], null, s26, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:1280
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_48
.LBB2_229:
	v_mad_co_i64_i32 v[15:16], null, s26, v13, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:2560
	global_load_b32 v17, v[15:16], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_49
.LBB2_230:
	v_mad_co_i64_i32 v[15:16], null, s26, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:3840
	global_load_b32 v17, v[15:16], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_50
.LBB2_231:
	v_mad_co_i64_i32 v[15:16], null, s26, v13, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:5120
	global_load_b32 v17, v[15:16], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_51
.LBB2_232:
	v_mad_co_i64_i32 v[15:16], null, s26, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:6400
	global_load_b32 v17, v[15:16], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_52
.LBB2_233:
	v_mad_co_i64_i32 v[15:16], null, s26, v13, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s5, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s5
	v_add_co_u32 v15, s5, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s5
	ds_load_b32 v18, v7 offset:7680
	global_load_b32 v17, v[15:16], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off offset:384
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_53
	s_branch .LBB2_54
.LBB2_234:
	v_mad_co_i64_i32 v[17:18], null, s26, v16, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:1280
	global_load_b32 v19, v[17:18], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execz .LBB2_58
.LBB2_235:
	v_mad_co_i64_i32 v[17:18], null, s26, v15, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:2560
	global_load_b32 v19, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_59
.LBB2_236:
	v_mad_co_i64_i32 v[17:18], null, s26, v16, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:3840
	global_load_b32 v19, v[17:18], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:128
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_60
.LBB2_237:
	v_mad_co_i64_i32 v[17:18], null, s26, v15, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:5120
	global_load_b32 v19, v[17:18], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_61
.LBB2_238:
	v_mad_co_i64_i32 v[17:18], null, s26, v16, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:6400
	global_load_b32 v19, v[17:18], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_62
.LBB2_239:
	v_mad_co_i64_i32 v[17:18], null, s26, v15, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, s7, s20, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, s7
	v_add_co_u32 v17, s7, v17, v19
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, s7
	ds_load_b32 v20, v7 offset:7680
	global_load_b32 v19, v[17:18], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:384
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_63
	s_branch .LBB2_64
.LBB2_240:
	v_mad_co_i64_i32 v[9:10], null, s26, v11, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7
	global_load_b32 v17, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_82
.LBB2_241:
	v_mad_co_i64_i32 v[9:10], null, s26, v12, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:1280
	global_load_b32 v17, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_83
.LBB2_242:
	v_mad_co_i64_i32 v[9:10], null, s26, v11, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:2560
	global_load_b32 v17, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_84
.LBB2_243:
	v_mad_co_i64_i32 v[9:10], null, s26, v12, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:3840
	global_load_b32 v17, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_85
.LBB2_244:
	v_mad_co_i64_i32 v[9:10], null, s26, v11, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:5120
	global_load_b32 v17, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB2_86
.LBB2_245:
	v_mad_co_i64_i32 v[9:10], null, s26, v12, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v18, v7 offset:6400
	global_load_b32 v17, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[9:10], v17, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_87
.LBB2_246:
	v_mad_co_i64_i32 v[9:10], null, s26, v11, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v18, vcc_lo
	ds_load_b32 v17, v7 offset:7680
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v17, v11
	global_store_b32 v[9:10], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_88
	s_branch .LBB2_89
.LBB2_247:
	v_mad_co_i64_i32 v[9:10], null, s26, v13, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_91
.LBB2_248:
	v_mad_co_i64_i32 v[9:10], null, s26, v14, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:1280
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_92
.LBB2_249:
	v_mad_co_i64_i32 v[9:10], null, s26, v13, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:2560
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_93
.LBB2_250:
	v_mad_co_i64_i32 v[9:10], null, s26, v14, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:3840
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_94
.LBB2_251:
	v_mad_co_i64_i32 v[9:10], null, s26, v13, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:5120
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_95
.LBB2_252:
	v_mad_co_i64_i32 v[9:10], null, s26, v14, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:6400
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_96
.LBB2_253:
	v_mad_co_i64_i32 v[9:10], null, s26, v13, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v7 offset:7680
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_97
	s_branch .LBB2_98
.LBB2_254:
	v_mad_co_i64_i32 v[8:9], null, s26, v15, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_100
.LBB2_255:
	v_mad_co_i64_i32 v[8:9], null, s26, v16, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:1280
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_101
.LBB2_256:
	v_mad_co_i64_i32 v[8:9], null, s26, v15, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:2560
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_102
.LBB2_257:
	v_mad_co_i64_i32 v[8:9], null, s26, v16, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:3840
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_103
.LBB2_258:
	v_mad_co_i64_i32 v[8:9], null, s26, v15, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:5120
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_104
.LBB2_259:
	v_mad_co_i64_i32 v[8:9], null, s26, v16, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:6400
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_105
.LBB2_260:
	v_mad_co_i64_i32 v[8:9], null, s26, v15, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v7 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_106
	s_branch .LBB2_107
.LBB2_261:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v4 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_146
.LBB2_262:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v4 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_147
.LBB2_263:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v4 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_148
.LBB2_264:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v4 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_149
.LBB2_265:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v4 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_150
.LBB2_266:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v4 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s3, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s3
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_151
	s_branch .LBB2_152
.LBB2_267:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v4 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_156
.LBB2_268:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v4 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_157
.LBB2_269:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v4 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_158
.LBB2_270:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v4 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_159
.LBB2_271:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v4 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_160
.LBB2_272:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v4 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s5, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s5
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_161
	s_branch .LBB2_162
.LBB2_273:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v4 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB2_166
.LBB2_274:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v4 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_167
.LBB2_275:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v4 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_168
.LBB2_276:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v4 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_169
.LBB2_277:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v4 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_170
.LBB2_278:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v4 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s7, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s7
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_171
	s_branch .LBB2_172
.LBB2_279:
	v_mad_co_i64_i32 v[6:7], null, s12, v8, 0
	ds_load_b32 v16, v4
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_190
.LBB2_280:
	v_mad_co_i64_i32 v[6:7], null, s12, v9, 0
	ds_load_b32 v16, v4 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_191
.LBB2_281:
	v_mad_co_i64_i32 v[6:7], null, s12, v8, 0
	ds_load_b32 v16, v4 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_192
.LBB2_282:
	v_mad_co_i64_i32 v[6:7], null, s12, v9, 0
	ds_load_b32 v16, v4 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_193
.LBB2_283:
	v_mad_co_i64_i32 v[6:7], null, s12, v8, 0
	ds_load_b32 v16, v4 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_194
.LBB2_284:
	v_mad_co_i64_i32 v[6:7], null, s12, v9, 0
	ds_load_b32 v16, v4 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_195
.LBB2_285:
	v_mad_co_i64_i32 v[6:7], null, s12, v8, 0
	ds_load_b32 v8, v4 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v8, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_196
	s_branch .LBB2_197
.LBB2_286:
	v_mad_co_i64_i32 v[6:7], null, s12, v10, 0
	ds_load_b32 v14, v4
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_199
.LBB2_287:
	v_mad_co_i64_i32 v[6:7], null, s12, v11, 0
	ds_load_b32 v14, v4 offset:1280
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_200
.LBB2_288:
	v_mad_co_i64_i32 v[6:7], null, s12, v10, 0
	ds_load_b32 v14, v4 offset:2560
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_201
.LBB2_289:
	v_mad_co_i64_i32 v[6:7], null, s12, v11, 0
	ds_load_b32 v14, v4 offset:3840
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_202
.LBB2_290:
	v_mad_co_i64_i32 v[6:7], null, s12, v10, 0
	ds_load_b32 v14, v4 offset:5120
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_203
.LBB2_291:
	v_mad_co_i64_i32 v[6:7], null, s12, v11, 0
	ds_load_b32 v14, v4 offset:6400
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_204
.LBB2_292:
	v_mad_co_i64_i32 v[6:7], null, s12, v10, 0
	ds_load_b32 v10, v4 offset:7680
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_205
	s_branch .LBB2_206
.LBB2_293:
	v_mad_co_i64_i32 v[5:6], null, s12, v12, 0
	ds_load_b32 v9, v4
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_208
.LBB2_294:
	v_mad_co_i64_i32 v[5:6], null, s12, v13, 0
	ds_load_b32 v9, v4 offset:1280
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_209
.LBB2_295:
	v_mad_co_i64_i32 v[5:6], null, s12, v12, 0
	ds_load_b32 v9, v4 offset:2560
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_210
.LBB2_296:
	v_mad_co_i64_i32 v[5:6], null, s12, v13, 0
	ds_load_b32 v9, v4 offset:3840
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_211
.LBB2_297:
	v_mad_co_i64_i32 v[5:6], null, s12, v12, 0
	ds_load_b32 v9, v4 offset:5120
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_212
.LBB2_298:
	v_mad_co_i64_i32 v[5:6], null, s12, v13, 0
	ds_load_b32 v9, v4 offset:6400
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_213
.LBB2_299:
	v_mad_co_i64_i32 v[5:6], null, s12, v12, 0
	ds_load_b32 v9, v4 offset:7680
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_214
	s_branch .LBB2_215
.Lfunc_end2:
	.size	gemm_mq4g256v2_residual_mmq_iu4, .Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 202
		.amdhsa_next_free_sgpr 41
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_vgpr, 202
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.numbered_sgpr, 41
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 34868
; TotalNumSgprs: 43
; NumVgprs: 202
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 202
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_full_add ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_full_add
	.globl	gemm_mq4g256v2_residual_mmq_iu4_full_add
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_full_add,@function
gemm_mq4g256v2_residual_mmq_iu4_full_add: ; @gemm_mq4g256v2_residual_mmq_iu4_full_add
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_108
; %bb.1:                                ; %.preheader559.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s3, s12, -1
	s_ashr_i32 s17, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s2, s17, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s18, v2
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v3, 2, v1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB3_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB3_4:                                ; %.preheader554.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s3, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s2, v5, s[4:5]
	global_load_b32 v5, v[5:6], off
	v_lshlrev_b32_e32 v6, 4, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v5, v6, v5
	v_cvt_f32_f16_e32 v5, v5.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_lshlrev_b32 v6, 3, v2
	v_add3_u32 v3, 0, v6, v3
	ds_store_2addr_stride64_b32 v3, v4, v5 offset0:48 offset1:56
.LBB3_5:                                ; %Flow791
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s18, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s16, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s3, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s3, v5
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s2, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s2, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v185, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v146, 0
	v_or_b32_e32 v15, 8, v185
	v_and_or_b32 v16, v185, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v180, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v152, 0 :: v_dual_and_b32 v159, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v141, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v184, v0, 4, 1
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_add_nc_u32_e32 v186, 0, v15
	v_add_nc_u32_e32 v187, 0, v11
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v176, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v8, 0, v8, s1
	v_cndmask_b32_e64 v7, 0, v7, s1
	ds_store_2addr_stride64_b64 v186, v[3:4], v[5:6] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[7:8], v[9:10] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_14
; %bb.6:                                ; %.preheader553.lr.ph.i
	v_dual_mov_b32 v176, 0 :: v_dual_add_nc_u32 v3, s16, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s3, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_dual_mov_b32 v183, 0 :: v_dual_add_nc_u32 v8, s18, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[65:66], null, s2, v7, s[4:5]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 6, v184
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v10, 2, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 8, v5
	v_lshl_or_b32 v189, v6, 11, v4
	v_lshl_or_b32 v190, v5, 10, v4
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v4, 9, v6
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v5, 3, v159
	v_mad_co_u64_u32 v[66:67], null, s2, v7, v[66:67]
	v_min_i32_e32 v188, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v191, 0, v2, v10
	v_cmp_gt_i32_e64 s3, s12, v3
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v192, 4, v1
	v_add3_u32 v193, 0, v11, v9
	v_add3_u32 v194, 0, v4, v5
	v_dual_mov_b32 v178, 0 :: v_dual_lshlrev_b32 v195, 2, v1
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s19, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_8:                                ; %.preheader553.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[22:23], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s17
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	s_mov_b32 s28, -1
	s_mov_b32 s29, 0
	s_branch .LBB3_10
.LBB3_9:                                ;   in Loop: Header=BB3_10 Depth=2
	s_wait_loadcnt_dscnt 0x30b
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v176, v176, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v183, v183, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v181, v181, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v182, v182, v57 :: v_dual_add_f32 v179, v179, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v180, v180, v59 :: v_dual_add_f32 v177, v177, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v178, v178, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_add_f32_e32 v175, v175, v49
	v_cvt_f32_i32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v60, v59 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_cvt_f32_i32_e32 v54, v54
	v_add_f32_e32 v174, v174, v50
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, v102, v92 :: v_dual_fmac_f32 v49, v52, v59
	v_mul_f32_e32 v50, v57, v53
	v_dual_mul_f32 v52, v96, v92 :: v_dual_mul_f32 v57, v98, v92
	v_cvt_f32_i32_e32 v53, v55
	v_dual_add_f32 v172, v172, v51 :: v_dual_mul_f32 v51, v58, v54
	v_dual_mul_f32 v55, v103, v92 :: v_dual_mul_f32 v54, v101, v92
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	v_dual_fmac_f32 v51, v55, v59 :: v_dual_fmac_f32 v50, v54, v59
	v_mul_f32_e32 v55, v99, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v54, v57, v56
	v_fmac_f32_e32 v52, v53, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v173, v173, v49 :: v_dual_add_f32 v170, v170, v51
	s_wait_dscnt 0x9
	v_mul_f32_e32 v49, v110, v90
	v_cvt_f32_i32_e32 v41, v41
	v_add_f32_e32 v171, v171, v50
	v_dual_fmac_f32 v54, v55, v59 :: v_dual_add_f32 v169, v169, v52
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v50, v108, v90 :: v_dual_mul_f32 v41, v49, v41
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v52, v109, v90
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v168, v168, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_mul_f32_e32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v166, v166, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v167, v167, v42 :: v_dual_add_f32 v164, v164, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v165, v165, v41 :: v_dual_add_f32 v162, v162, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v163, v163, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v160, v160, v44 :: v_dual_add_f32 v161, v161, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_dual_mul_f32 v42, v107, v72 :: v_dual_add_f32 v157, v157, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v44, v43
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v33, v41, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v36, v105, v72 :: v_dual_fmac_f32 v35, v42, v43
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_dual_add_f32 v158, v158, v34 :: v_dual_fmac_f32 v33, v36, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v155, v155, v35 :: v_dual_mul_f32 v34, v41, v37
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v37, v103, v72 :: v_dual_mul_f32 v38, v96, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v98, v72 :: v_dual_add_f32 v156, v156, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_mul_f32 v38, v99, v72 :: v_dual_add_f32 v153, v153, v34
	v_fmac_f32_e32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v154, v154, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v151, v151, v33
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v149, v149, v25 :: v_dual_add_f32 v150, v150, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v147, v147, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v148, v148, v27
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v146, v146, v28
	v_mul_f32_e32 v28, v94, v75
	v_fmac_f32_e32 v25, v26, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	v_dual_add_f32 v145, v145, v25 :: v_dual_fmac_f32 v26, v28, v95
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v28, v92, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v92, v84
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v143, v143, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v19, v29, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v25, v92, v83 :: v_dual_add_f32 v142, v142, v18
	v_dual_add_f32 v144, v144, v27 :: v_dual_add_f32 v141, v141, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_dual_add_f32 v139, v139, v20 :: v_dual_mul_f32 v20, v92, v74
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_dual_add_f32 v140, v140, v19 :: v_dual_mul_f32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v13, v13
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v2, v2
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v10, v22, v10 :: v_dual_add_f32 v137, v137, v17
	v_add_f32_e32 v138, v138, v19
	v_mul_f32_e32 v19, v90, v87
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v10, v19, v51
	v_dual_add_f32 v135, v135, v20 :: v_dual_mul_f32 v20, v90, v82
	v_dual_mul_f32 v12, v21, v12 :: v_dual_fmac_f32 v17, v18, v59
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v11, v20, v11 :: v_dual_mul_f32 v20, v90, v78
	v_add_f32_e32 v134, v134, v10
	v_fmac_f32_e32 v9, v18, v51
	v_mul_f32_e32 v18, v90, v85
	v_mul_f32_e32 v10, v90, v74
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v133, v133, v9
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v132, v132, v12
	v_dual_add_f32 v136, v136, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	v_cvt_f32_i32_e32 v8, v8
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_add_f32_e32 v131, v131, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	v_add_f32_e32 v152, v152, v36
	s_xor_b32 s8, s28, -1
	s_mov_b32 s30, 1
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	v_mul_f32_e32 v10, v90, v75
	s_mov_b32 s29, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_mul_f32_e32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v1, v12, v43
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v125, v125, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v127, v127, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v126, v126, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v128, v128, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v122, v122, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v123, v123, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v120, v120, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v121, v121, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_7
.LBB3_10:                               ;   Parent Loop BB3_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s30, s11
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[8:9], s[14:15]
	s_mov_b32 s35, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[6:7], s[36:37]
	v_add_nc_u32_e32 v80, s26, v190
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	v_add_co_u32 v3, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	v_add_co_u32 v5, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v7, s31, s34, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s31
	global_load_b64 v[118:119], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[116:117], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	v_add_nc_u32_e32 v81, s27, v189
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v81 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v81 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v80 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v81 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v81 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s0
	v_cndmask_b32_e64 v72, 0, v118, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_wait_loadcnt 0x2
	ds_store_2addr_stride64_b64 v186, v[112:113], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s1
	v_cndmask_b32_e64 v72, 0, v116, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[114:115], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.11:                               ;   in Loop: Header=BB3_10 Depth=2
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s30, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[4:5], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s9
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s8, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[118:119], v[72:73], off offset:8
	global_load_b64 v[116:117], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB3_12:                               ; %.preheader551.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	v_add_nc_u32_e32 v85, 0, v189
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s19, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v84
	v_add_nc_u32_e32 v80, 0x100, v85
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s27, v193
	v_add_nc_u32_e32 v72, s26, v194
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_9
; %bb.13:                               ; %.preheader552.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s8, s19, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s8, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB3_9
.LBB3_14:                               ; %._crit_edge623.i
	v_mul_u32_u24_e32 v1, 0x500, v185
	v_lshlrev_b32_e32 v2, 2, v159
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v5, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v3, 0x50, v159
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x280, v184, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_105
; %bb.15:                               ; %.preheader548.i
	ds_store_2addr_b32 v2, v176, v183 offset1:20
	ds_store_2addr_b32 v2, v181, v182 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v179, v180 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v177, v178 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v159
	v_or_b32_e32 v8, s18, v5
	v_add3_u32 v6, 0, v3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	ds_load_b32 v12, v6
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off
.LBB3_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:1280
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off
.LBB3_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:2560
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
.LBB3_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:3840
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
.LBB3_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:5120
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
.LBB3_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:6400
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
.LBB3_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:7680
	global_load_b32 v12, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:384
.LBB3_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v184
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v6 offset:8960
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384
.LBB3_31:                               ; %.preheader546.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v175, v174 offset1:20
	ds_store_2addr_b32 v7, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s2, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v14, v6
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off
.LBB3_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_110
.LBB3_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_111
.LBB3_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_112
.LBB3_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_113
.LBB3_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_114
.LBB3_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_41
.LBB3_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:8960
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
.LBB3_41:                               ; %.preheader546.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v166, v167 offset1:20
	ds_store_2addr_b32 v7, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	ds_load_b32 v16, v6
	global_load_b32 v15, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off
.LBB3_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_116
.LBB3_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_117
.LBB3_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_118
.LBB3_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_119
.LBB3_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_120
.LBB3_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_51
.LBB3_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384
.LBB3_51:                               ; %.preheader546.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v157, v158 offset1:20
	ds_store_2addr_b32 v7, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB3_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	ds_load_b32 v18, v6
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off
.LBB3_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_121
; %bb.54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_122
.LBB3_55:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_123
.LBB3_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_124
.LBB3_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_125
.LBB3_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_126
.LBB3_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_61
.LBB3_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:8960
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384
.LBB3_61:                               ; %.preheader547.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v149, v150 offset1:20
	ds_store_2addr_b32 v7, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64
.LBB3_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64
.LBB3_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192
.LBB3_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192
.LBB3_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320
.LBB3_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320
.LBB3_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB3_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v8, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v8, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	ds_load_b32 v18, v6 offset:7680
	global_load_b32 v8, v[16:17], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v18, v8
	global_store_b32 v[16:17], v8, off offset:448
.LBB3_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:448
.LBB3_77:                               ; %.preheader546.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v141, v142 offset1:20
	ds_store_2addr_b32 v7, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_128
.LBB3_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_129
.LBB3_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_130
.LBB3_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_131
.LBB3_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_132
.LBB3_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_133
.LBB3_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_86
.LBB3_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
.LBB3_86:                               ; %.preheader546.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v133, v134 offset1:20
	ds_store_2addr_b32 v7, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_135
.LBB3_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_136
.LBB3_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_137
.LBB3_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_138
.LBB3_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_139
.LBB3_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_140
.LBB3_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_95
.LBB3_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
.LBB3_95:                               ; %.preheader546.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v125, v126 offset1:20
	ds_store_2addr_b32 v7, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_142
.LBB3_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_143
.LBB3_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_144
.LBB3_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_145
.LBB3_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_146
.LBB3_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_147
.LBB3_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_104
.LBB3_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	ds_load_b32 v6, v6 offset:8960
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v7, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v8, v1, vcc_lo
	global_load_b32 v7, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	global_store_b32 v[0:1], v6, off offset:448
.LBB3_104:                              ; %.loopexit.loopexit656.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB3_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_107
; %bb.106:                              ; %.preheader545.i
	ds_store_2addr_b32 v2, v176, v183 offset1:20
	ds_store_2addr_b32 v2, v181, v182 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v179, v180 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v177, v178 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_lo_u32 v0, s12, v5
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[2:3], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v3, 0, v3, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v5, v0, v159, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, -1
	s_lshl_b32 s5, s12, 8
	s_movk_i32 s4, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v6, v5, s[0:3], null offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_or_b32 s6, s5, 0x80
	s_mul_i32 s7, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s7, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v6
	buffer_store_b32 v0, v5, s[0:3], null offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s6, s5, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s6, s5, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_lshl_b32 s4, s12, 6
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v175, v174 offset1:20
	ds_store_2addr_b32 v2, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s8, s7, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s8, s7, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_lshl_b32 s6, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s6, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s8, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v166, v167 offset1:20
	ds_store_2addr_b32 v2, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s10, s8, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s10, s8, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_mul_i32 s9, s12, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s12, s10, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v157, v158 offset1:20
	ds_store_2addr_b32 v2, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s10, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s12, s10, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_mov_b32 s11, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	s_or_b32 s12, s5, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v149, v150 offset1:20
	ds_store_2addr_b32 v2, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_or_b32 s12, s5, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s11, s7, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s5, s4, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v141, v142 offset1:20
	ds_store_2addr_b32 v2, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s11, s7, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s11, s7, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_or_b32 s5, s8, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_or_b32 s4, s6, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v133, v134 offset1:20
	ds_store_2addr_b32 v2, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s8, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0x140
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s8, 0x140
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s5, s10, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v125, v126 offset1:20
	ds_store_2addr_b32 v2, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s10, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s10, 0x140
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB3_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB3_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1ELb0ELb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB3_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:1280
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_35
.LBB3_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:2560
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_36
.LBB3_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:3840
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_37
.LBB3_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:5120
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_38
.LBB3_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:6400
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_39
.LBB3_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:7680
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_40
	s_branch .LBB3_41
.LBB3_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_45
.LBB3_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_46
.LBB3_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_47
.LBB3_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_48
.LBB3_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_49
.LBB3_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:7680
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_50
	s_branch .LBB3_51
.LBB3_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB3_55
.LBB3_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_56
.LBB3_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_57
.LBB3_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_58
.LBB3_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_59
.LBB3_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:7680
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_60
	s_branch .LBB3_61
.LBB3_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_79
.LBB3_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_80
.LBB3_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_81
.LBB3_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_82
.LBB3_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_83
.LBB3_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_84
.LBB3_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v16, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v16, v10
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_85
	s_branch .LBB3_86
.LBB3_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_88
.LBB3_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:1280
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_89
.LBB3_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:2560
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_90
.LBB3_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:3840
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
.LBB3_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:5120
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_92
.LBB3_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:6400
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_93
.LBB3_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_94
	s_branch .LBB3_95
.LBB3_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_97
.LBB3_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:1280
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_98
.LBB3_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:2560
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_99
.LBB3_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:3840
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_100
.LBB3_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:5120
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_101
.LBB3_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:6400
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_102
.LBB3_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:7680
	global_load_b32 v9, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_103
	s_branch .LBB3_104
.Lfunc_end3:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_add, .Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 202
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_add)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_vgpr, 202
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18592
; TotalNumSgprs: 40
; NumVgprs: 202
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 202
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_full_set ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_full_set
	.globl	gemm_mq4g256v2_residual_mmq_iu4_full_set
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_full_set,@function
gemm_mq4g256v2_residual_mmq_iu4_full_set: ; @gemm_mq4g256v2_residual_mmq_iu4_full_set
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB4_108
; %bb.1:                                ; %.preheader552.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s3, s12, -1
	s_ashr_i32 s17, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s2, s17, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB4_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s18, v2
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v3, 2, v1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB4_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB4_4:                                ; %.preheader547.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s16, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s3, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s2, v5, s[4:5]
	global_load_b32 v5, v[5:6], off
	v_lshlrev_b32_e32 v6, 4, v1
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshrrev_b32_e32 v5, v6, v5
	v_cvt_f32_f16_e32 v5, v5.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v5, 0, v5 :: v_dual_lshlrev_b32 v6, 3, v2
	v_add3_u32 v3, 0, v6, v3
	ds_store_2addr_stride64_b32 v3, v4, v5 offset0:48 offset1:56
.LBB4_5:                                ; %Flow791
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s18, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s16, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s3, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s3, v5
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s2, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s2, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v185, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v146, 0
	v_or_b32_e32 v15, 8, v185
	v_and_or_b32 v16, v185, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v180, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v152, 0 :: v_dual_and_b32 v147, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v141, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v184, v0, 4, 1
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_add_nc_u32_e32 v186, 0, v15
	v_add_nc_u32_e32 v187, 0, v11
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v176, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v8, 0, v8, s1
	v_cndmask_b32_e64 v7, 0, v7, s1
	ds_store_2addr_stride64_b64 v186, v[3:4], v[5:6] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[7:8], v[9:10] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_14
; %bb.6:                                ; %.preheader546.lr.ph.i
	v_dual_mov_b32 v176, 0 :: v_dual_add_nc_u32 v3, s16, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s3, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_dual_mov_b32 v183, 0 :: v_dual_add_nc_u32 v8, s18, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[65:66], null, s2, v7, s[4:5]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 6, v184
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v10, 2, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 8, v5
	v_lshl_or_b32 v189, v6, 11, v4
	v_lshl_or_b32 v190, v5, 10, v4
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v4, 9, v6
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v5, 3, v147
	v_mad_co_u64_u32 v[66:67], null, s2, v7, v[66:67]
	v_min_i32_e32 v188, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v191, 0, v2, v10
	v_cmp_gt_i32_e64 s3, s12, v3
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v192, 4, v1
	v_add3_u32 v193, 0, v11, v9
	v_add3_u32 v194, 0, v4, v5
	v_dual_mov_b32 v178, 0 :: v_dual_lshlrev_b32 v195, 2, v1
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s19, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	s_branch .LBB4_8
.LBB4_7:                                ;   in Loop: Header=BB4_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_14
.LBB4_8:                                ; %.preheader546.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB4_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[22:23], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s17
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	s_mov_b32 s28, -1
	s_mov_b32 s29, 0
	s_branch .LBB4_10
.LBB4_9:                                ;   in Loop: Header=BB4_10 Depth=2
	s_wait_loadcnt_dscnt 0x30b
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v176, v176, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v183, v183, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v181, v181, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v182, v182, v57 :: v_dual_add_f32 v179, v179, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v180, v180, v59 :: v_dual_add_f32 v177, v177, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v178, v178, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_add_f32_e32 v175, v175, v49
	v_cvt_f32_i32_e32 v53, v53
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v60, v59 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_cvt_f32_i32_e32 v54, v54
	v_add_f32_e32 v174, v174, v50
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, v102, v92 :: v_dual_fmac_f32 v49, v52, v59
	v_mul_f32_e32 v50, v57, v53
	v_dual_mul_f32 v52, v96, v92 :: v_dual_mul_f32 v57, v98, v92
	v_cvt_f32_i32_e32 v53, v55
	v_dual_add_f32 v172, v172, v51 :: v_dual_mul_f32 v51, v58, v54
	v_dual_mul_f32 v55, v103, v92 :: v_dual_mul_f32 v54, v101, v92
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	v_dual_fmac_f32 v51, v55, v59 :: v_dual_fmac_f32 v50, v54, v59
	v_mul_f32_e32 v55, v99, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v54, v57, v56
	v_fmac_f32_e32 v52, v53, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v173, v173, v49 :: v_dual_add_f32 v170, v170, v51
	s_wait_dscnt 0x9
	v_mul_f32_e32 v49, v110, v90
	v_cvt_f32_i32_e32 v41, v41
	v_add_f32_e32 v171, v171, v50
	v_dual_fmac_f32 v54, v55, v59 :: v_dual_add_f32 v169, v169, v52
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v50, v108, v90 :: v_dual_mul_f32 v41, v49, v41
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v52, v109, v90
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v168, v168, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_mul_f32_e32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v166, v166, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v167, v167, v42 :: v_dual_add_f32 v164, v164, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v165, v165, v41 :: v_dual_add_f32 v162, v162, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v163, v163, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v160, v160, v44 :: v_dual_add_f32 v161, v161, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v42, v107, v72
	v_cvt_f32_i32_e32 v37, v37
	v_fmac_f32_e32 v34, v44, v43
	v_dual_add_f32 v158, v158, v33 :: v_dual_mul_f32 v33, v41, v36
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v36, v105, v72
	v_fmac_f32_e32 v35, v42, v43
	v_mul_f32_e32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_add_f32_e32 v159, v159, v34
	v_fmac_f32_e32 v33, v36, v43
	v_add_f32_e32 v156, v156, v35
	v_dual_mul_f32 v34, v41, v37 :: v_dual_mul_f32 v37, v103, v72
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v41, v98, v72 :: v_dual_mul_f32 v38, v96, v72
	v_add_f32_e32 v157, v157, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	v_mul_f32_e32 v38, v99, v72
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_add_f32 v154, v154, v34 :: v_dual_fmac_f32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_mul_f32 v25, v39, v25
	v_dual_mul_f32 v26, v40, v26 :: v_dual_mul_f32 v37, v94, v87
	v_dual_mul_f32 v34, v94, v89 :: v_dual_add_f32 v155, v155, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v152, v152, v33 :: v_dual_add_f32 v153, v153, v36
	v_dual_fmac_f32 v26, v37, v95 :: v_dual_fmac_f32 v25, v34, v95
	s_wait_dscnt 0x4
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v150, v150, v25 :: v_dual_add_f32 v151, v151, v26
	v_mul_f32_e32 v26, v94, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v28, v29, v95
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v148, v148, v25 :: v_dual_mul_f32 v25, v26, v30
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_dual_mul_f32 v30, v94, v76 :: v_dual_add_f32 v149, v149, v27
	v_add_f32_e32 v146, v146, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v26, v95
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v28, v94, v75 :: v_dual_mul_f32 v29, v94, v77
	v_dual_mul_f32 v30, v92, v88 :: v_dual_mul_f32 v31, v92, v86
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v145, v145, v25 :: v_dual_fmac_f32 v26, v28, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v30, v92, v84
	v_mul_f32_e32 v28, v92, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v27, v29, v95 :: v_dual_mul_f32 v18, v31, v18
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v143, v143, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v19, v29, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v25, v92, v83 :: v_dual_add_f32 v142, v142, v18
	v_dual_add_f32 v144, v144, v27 :: v_dual_add_f32 v141, v141, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_dual_add_f32 v139, v139, v20 :: v_dual_mul_f32 v20, v92, v74
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_dual_add_f32 v140, v140, v19 :: v_dual_mul_f32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v13, v13
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v2, v2
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v10, v22, v10 :: v_dual_add_f32 v137, v137, v17
	v_add_f32_e32 v138, v138, v19
	v_mul_f32_e32 v19, v90, v87
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v10, v19, v51
	v_dual_add_f32 v135, v135, v20 :: v_dual_mul_f32 v20, v90, v82
	v_dual_mul_f32 v12, v21, v12 :: v_dual_fmac_f32 v17, v18, v59
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v11, v20, v11 :: v_dual_mul_f32 v20, v90, v78
	v_add_f32_e32 v134, v134, v10
	v_fmac_f32_e32 v9, v18, v51
	v_mul_f32_e32 v18, v90, v85
	v_mul_f32_e32 v10, v90, v74
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v133, v133, v9
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v132, v132, v12
	v_dual_add_f32 v136, v136, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	v_cvt_f32_i32_e32 v8, v8
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	v_add_f32_e32 v131, v131, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	s_xor_b32 s8, s28, -1
	s_mov_b32 s30, 1
	s_mov_b32 s28, 0
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s29, -1
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	v_mul_f32_e32 v10, v90, v75
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_mul_f32_e32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v1, v12, v43
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v125, v125, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v127, v127, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v126, v126, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v128, v128, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v122, v122, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v123, v123, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v120, v120, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v121, v121, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_7
.LBB4_10:                               ;   Parent Loop BB4_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s30, s11
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[8:9], s[14:15]
	s_mov_b32 s35, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[6:7], s[36:37]
	v_add_nc_u32_e32 v80, s26, v190
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	v_add_co_u32 v3, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	v_add_co_u32 v5, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v7, s31, s34, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s31
	global_load_b64 v[118:119], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[116:117], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	v_add_nc_u32_e32 v81, s27, v189
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v81 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v81 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v80 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v81 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v81 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s0
	v_cndmask_b32_e64 v72, 0, v118, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_wait_loadcnt 0x2
	ds_store_2addr_stride64_b64 v186, v[112:113], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s1
	v_cndmask_b32_e64 v72, 0, v116, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v187, v[114:115], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_12
; %bb.11:                               ;   in Loop: Header=BB4_10 Depth=2
	s_add_co_i32 s8, s8, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s30, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[4:5], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s9
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s8, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[118:119], v[72:73], off offset:8
	global_load_b64 v[116:117], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB4_12:                               ; %.preheader544.i
                                        ;   in Loop: Header=BB4_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	v_add_nc_u32_e32 v85, 0, v189
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s19, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v84
	v_add_nc_u32_e32 v80, 0x100, v85
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s27, v193
	v_add_nc_u32_e32 v72, s26, v194
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_9
; %bb.13:                               ; %.preheader545.i
                                        ;   in Loop: Header=BB4_10 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s8, s19, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s8, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB4_9
.LBB4_14:                               ; %._crit_edge616.i
	v_mul_u32_u24_e32 v1, 0x500, v185
	v_lshlrev_b32_e32 v2, 2, v147
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v3, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v4, 0x50, v147
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v3
	v_mad_u32_u24 v2, 0x280, v184, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_105
; %bb.15:                               ; %.preheader541.i
	ds_store_2addr_b32 v2, v176, v183 offset1:20
	ds_store_2addr_b32 v2, v181, v182 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v179, v180 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v177, v178 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v147
	v_or_b32_e32 v8, s18, v3
	v_add3_u32 v6, 0, v4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB4_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v6
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB4_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
.LBB4_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB4_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB4_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB4_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB4_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB4_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v184
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB4_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	ds_load_b32 v15, v6 offset:8960
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:384
.LBB4_31:                               ; %.preheader539.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v175, v174 offset1:20
	ds_store_2addr_b32 v7, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB4_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v6
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, s20, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB4_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_110
.LBB4_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_111
.LBB4_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_112
.LBB4_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_113
.LBB4_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_114
.LBB4_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_41
.LBB4_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
.LBB4_41:                               ; %.preheader539.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v166, v167 offset1:20
	ds_store_2addr_b32 v7, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB4_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	ds_load_b32 v17, v6
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v17, off
.LBB4_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_116
.LBB4_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_117
.LBB4_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_118
.LBB4_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_119
.LBB4_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_120
.LBB4_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_51
.LBB4_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
.LBB4_51:                               ; %.preheader539.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v158, v159 offset1:20
	ds_store_2addr_b32 v7, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB4_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	ds_load_b32 v19, v6
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v19, off
.LBB4_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB4_121
; %bb.54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB4_122
.LBB4_55:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB4_123
.LBB4_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB4_124
.LBB4_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB4_125
.LBB4_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB4_126
.LBB4_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB4_61
.LBB4_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:8960
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384
.LBB4_61:                               ; %.preheader540.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v150, v151 offset1:20
	ds_store_2addr_b32 v7, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB4_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64
.LBB4_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB4_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64
.LBB4_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB4_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192
.LBB4_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB4_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192
.LBB4_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB4_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v16, s10, s20, v16
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320
.LBB4_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB4_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320
.LBB4_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB4_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v8, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v8, off offset:448
.LBB4_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:448
.LBB4_77:                               ; %.preheader539.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v141, v142 offset1:20
	ds_store_2addr_b32 v7, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_128
.LBB4_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_129
.LBB4_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_130
.LBB4_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_131
.LBB4_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB4_132
.LBB4_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_133
.LBB4_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_86
.LBB4_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:448
.LBB4_86:                               ; %.preheader539.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v133, v134 offset1:20
	ds_store_2addr_b32 v7, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_135
.LBB4_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_136
.LBB4_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_137
.LBB4_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_138
.LBB4_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_139
.LBB4_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_140
.LBB4_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_95
.LBB4_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v12, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448
.LBB4_95:                               ; %.preheader539.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v125, v126 offset1:20
	ds_store_2addr_b32 v7, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_142
.LBB4_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_143
.LBB4_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_144
.LBB4_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_145
.LBB4_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_146
.LBB4_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_147
.LBB4_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_104
.LBB4_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v9, v6 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v7, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v9, off offset:448
.LBB4_104:                              ; %.loopexit.loopexit649.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB4_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB4_107
; %bb.106:                              ; %.preheader538.i
	ds_store_2addr_b32 v2, v176, v183 offset1:20
	ds_store_2addr_b32 v2, v181, v182 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v179, v180 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v177, v178 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add3_u32 v10, 0, v4, v5
	v_mul_lo_u32 v3, s12, v3
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[4:5], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v11, v3, v147, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_mov_b32 s2, -1
	s_lshl_b32 s6, s12, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_movk_i32 s7, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[8:9], v10 offset0:30 offset1:35
	s_movk_i32 s8, 0x100
	s_movk_i32 s9, 0x180
	s_or_b32 s10, s6, 0x80
	s_add_co_i32 s4, s6, 0x100
	s_add_co_i32 s5, s6, 0x180
	s_add_co_i32 s17, s6, 0x140
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], null offen
	buffer_store_b32 v1, v11, s[0:3], s6 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s7 offen
	buffer_store_b32 v5, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s8 offen
	buffer_store_b32 v7, v11, s[0:3], s4 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v8, v11, s[0:3], s9 offen
	buffer_store_b32 v9, v11, s[0:3], s5 offen
	s_lshl_b32 s4, s12, 6
	s_mul_i32 s5, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 0x80
	s_add_co_i32 s8, s5, 0x80
	s_add_co_i32 s9, s4, 0x100
	s_add_co_i32 s10, s5, 0x100
	s_add_co_i32 s11, s4, 0x180
	s_add_co_i32 s13, s5, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v175, v174 offset1:20
	ds_store_2addr_b32 v2, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	buffer_store_b32 v4, v11, s[0:3], s8 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s9 offen
	buffer_store_b32 v6, v11, s[0:3], s10 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s11 offen
	buffer_store_b32 v8, v11, s[0:3], s13 offen
	s_lshl_b32 s7, s12, 7
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s7, 0x80
	s_add_co_i32 s10, s8, 0x80
	s_add_co_i32 s11, s7, 0x100
	s_add_co_i32 s13, s8, 0x100
	s_add_co_i32 s14, s7, 0x180
	s_add_co_i32 s15, s8, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v166, v167 offset1:20
	ds_store_2addr_b32 v2, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s7 offen
	buffer_store_b32 v1, v11, s[0:3], s8 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s9 offen
	buffer_store_b32 v4, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	buffer_store_b32 v6, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s14 offen
	buffer_store_b32 v8, v11, s[0:3], s15 offen
	s_mul_i32 s9, s12, 0xc0
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_add_co_i32 s12, s10, 0x80
	s_add_co_i32 s13, s9, 0x100
	s_add_co_i32 s14, s10, 0x100
	s_add_co_i32 s15, s9, 0x180
	s_add_co_i32 s16, s10, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v158, v159 offset1:20
	ds_store_2addr_b32 v2, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s9 offen
	buffer_store_b32 v1, v11, s[0:3], s10 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s11 offen
	buffer_store_b32 v4, v11, s[0:3], s12 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s13 offen
	buffer_store_b32 v6, v11, s[0:3], s14 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s15 offen
	buffer_store_b32 v8, v11, s[0:3], s16 offen
	s_mov_b32 s14, 64
	s_or_b32 s15, s6, 64
	s_movk_i32 s11, 0x140
	s_movk_i32 s12, 0xc0
	s_movk_i32 s13, 0x1c0
	s_or_b32 s16, s6, 0xc0
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v150, v151 offset1:20
	ds_store_2addr_b32 v2, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s14 offen
	buffer_store_b32 v1, v11, s[0:3], s15 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen
	buffer_store_b32 v4, v11, s[0:3], s16 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	buffer_store_b32 v6, v11, s[0:3], s17 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s13 offen
	buffer_store_b32 v8, v11, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 64
	s_add_co_i32 s11, s5, 64
	s_add_co_i32 s12, s4, 0xc0
	s_add_co_i32 s13, s5, 0xc0
	s_add_co_i32 s14, s4, 0x140
	s_add_co_i32 s15, s5, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v141, v142 offset1:20
	ds_store_2addr_b32 v2, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s6 offen
	buffer_store_b32 v1, v11, s[0:3], s11 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen
	buffer_store_b32 v4, v11, s[0:3], s13 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s14 offen
	buffer_store_b32 v6, v11, s[0:3], s15 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s4 offen
	buffer_store_b32 v8, v11, s[0:3], s5 offen
	s_or_b32 s4, s7, 64
	s_or_b32 s5, s8, 64
	s_add_co_i32 s6, s7, 0xc0
	s_add_co_i32 s11, s8, 0xc0
	s_add_co_i32 s12, s7, 0x140
	s_add_co_i32 s13, s8, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v133, v134 offset1:20
	ds_store_2addr_b32 v2, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s6 offen
	buffer_store_b32 v4, v11, s[0:3], s11 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s12 offen
	buffer_store_b32 v6, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s7 offen
	buffer_store_b32 v8, v11, s[0:3], s8 offen
	s_add_co_i32 s4, s9, 64
	s_add_co_i32 s5, s10, 64
	s_add_co_i32 s6, s9, 0xc0
	s_add_co_i32 s7, s10, 0xc0
	s_add_co_i32 s8, s9, 0x140
	s_add_co_i32 s11, s10, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v125, v126 offset1:20
	ds_store_2addr_b32 v2, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s6 offen
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s8 offen
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s9 offen
	buffer_store_b32 v7, v11, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB4_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB4_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0ELb0ELb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB4_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_35
.LBB4_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_36
.LBB4_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_37
.LBB4_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_38
.LBB4_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB4_39
.LBB4_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB4_40
	s_branch .LBB4_41
.LBB4_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_45
.LBB4_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_46
.LBB4_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_47
.LBB4_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_48
.LBB4_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB4_49
.LBB4_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB4_50
	s_branch .LBB4_51
.LBB4_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB4_55
.LBB4_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB4_56
.LBB4_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB4_57
.LBB4_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB4_58
.LBB4_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB4_59
.LBB4_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB4_60
	s_branch .LBB4_61
.LBB4_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_79
.LBB4_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_80
.LBB4_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_81
.LBB4_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_82
.LBB4_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB4_83
.LBB4_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_84
.LBB4_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v10, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_85
	s_branch .LBB4_86
.LBB4_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_88
.LBB4_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_89
.LBB4_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_90
.LBB4_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_91
.LBB4_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_92
.LBB4_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_93
.LBB4_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v12, v6 offset:7680
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_94
	s_branch .LBB4_95
.LBB4_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_97
.LBB4_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:1280
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_98
.LBB4_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:2560
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_99
.LBB4_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:3840
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_100
.LBB4_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:5120
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_101
.LBB4_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:6400
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB4_102
.LBB4_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:7680
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB4_103
	s_branch .LBB4_104
.Lfunc_end4:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_set, .Lfunc_end4-gemm_mq4g256v2_residual_mmq_iu4_full_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 202
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-gemm_mq4g256v2_residual_mmq_iu4_full_set)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_vgpr, 202
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16304
; TotalNumSgprs: 40
; NumVgprs: 202
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 202
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_atiled_add ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_atiled_add
	.globl	gemm_mq4g256v2_residual_mmq_iu4_atiled_add
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_atiled_add,@function
gemm_mq4g256v2_residual_mmq_iu4_atiled_add: ; @gemm_mq4g256v2_residual_mmq_iu4_atiled_add
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB5_108
; %bb.1:                                ; %.preheader508.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_add_co_i32 s2, s14, 0x7f
	s_ashr_i32 s1, s13, 31
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s10, s1, 24
	s_lshr_b32 s0, s3, 25
	s_add_co_i32 s10, s13, s10
	s_add_co_i32 s2, s2, s0
	s_lshr_b32 s0, s1, 26
	s_ashr_i32 s2, s2, 7
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s0, s0, 6
	s_lshl_b64 s[8:9], s[2:3], 12
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v3, 1, v0
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	s_ashr_i32 s15, s10, 8
	s_add_co_i32 s17, s12, -1
	s_mul_i32 s10, s15, 0x88
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB5_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s18, v4
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v3
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v1
	s_cbranch_execz .LBB5_4
; %bb.3:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 3, v[1:2]
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB5_4:                                ; %.preheader503.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v1, s16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v6, s17, v1
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mad_co_i64_i32 v[6:7], null, s10, v6, s[4:5]
	global_load_b32 v6, v[6:7], off
	v_lshlrev_b32_e32 v7, 4, v3
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v6, v7, v6
	v_lshlrev_b32_e32 v7, 3, v4
	v_cvt_f32_f16_e32 v6, v6.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v5, 0, v7, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v6, vcc_lo
	ds_store_2addr_stride64_b32 v5, v2, v1 offset0:32 offset1:40
.LBB5_5:                                ; %Flow821
	s_or_b32 exec_lo, exec_lo, s11
	v_lshrrev_b32_e32 v7, 2, v0
	v_dual_mov_b32 v116, 0 :: v_dual_and_b32 v1, 3, v0
	v_dual_mov_b32 v148, 0 :: v_dual_lshlrev_b32 v9, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v2, s16, v7
	v_dual_mov_b32 v120, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v168, 0 :: v_dual_and_b32 v9, 16, v9
	v_dual_mov_b32 v118, 0 :: v_dual_add_nc_u32 v5, 64, v2
	v_min_i32_e32 v2, s17, v2
	v_lshrrev_b32_e32 v181, 5, v0
	v_bfe_u32 v8, v0, 1, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v5, s17, v5
	v_and_or_b32 v7, v7, 15, v9
	v_mad_co_u64_u32 v[68:69], null, s10, v2, v[1:2]
	v_or_b32_e32 v9, 8, v181
	v_and_or_b32 v10, v181, 6, v8
	v_mad_co_u64_u32 v[69:70], null, s10, v5, v[1:2]
	v_dual_mov_b32 v142, 0 :: v_dual_lshlrev_b32 v7, 3, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v8, v9, 14, v8
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v155, 15, v0
	v_mov_b32_e32 v122, 0
	s_clause 0x1
	global_load_b64 v[1:2], v68, s[4:5] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	v_lshl_or_b32 v9, v10, 8, v7
	v_lshl_or_b32 v7, v8, 8, v7
	v_mov_b32_e32 v137, 0
	v_bfe_u32 v180, v0, 4, 1
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_add_nc_u32_e32 v182, 0, v9
	v_add_nc_u32_e32 v183, 0, v7
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s11, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x1
	ds_store_b64 v182, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b64 v183, v[5:6]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_14
; %bb.6:                                ; %.preheader502.lr.ph.i
	v_dual_mov_b32 v172, 0 :: v_dual_and_b32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 6, v0
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v7, s16, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v194, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v193, 0 :: v_dual_add_nc_u32 v6, s18, v4
	v_dual_mov_b32 v175, 0 :: v_dual_lshlrev_b32 v10, 2, v3
	v_lshl_or_b32 v9, v5, 11, v1
	v_lshl_or_b32 v184, v2, 10, v1
	v_min_i32_e32 v1, s17, v7
	v_dual_mov_b32 v173, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	s_add_co_i32 s13, s14, -1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s10, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v70, s13, v6
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v2, 6, v180
	v_add3_u32 v187, 0, v4, v10
	v_dual_mov_b32 v145, 0 :: v_dual_lshlrev_b32 v4, 3, v155
	v_mad_co_u64_u32 v[66:67], null, s10, v1, v[66:67]
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v1, 9, v5
	v_add_co_u32 v185, s0, s0, v10
	v_add_co_u32 v190, s6, s6, v9
	v_ashrrev_i32_e32 v71, 31, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v186, null, s1, 0, s0
	v_cmp_gt_i32_e64 s0, s14, v6
	v_cmp_gt_i32_e64 s1, s12, v7
	v_dual_mov_b32 v143, 0 :: v_dual_lshlrev_b32 v188, 4, v3
	v_add3_u32 v189, 0, v8, v2
	v_add_co_ci_u32_e64 v191, null, s7, 0, s6
	v_add3_u32 v192, 0, v1, v4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	s_mov_b32 s22, ttmp7
	s_mov_b32 s23, s11
	s_ashr_i32 s13, s14, 31
	s_movk_i32 s17, 0x2000
	s_movk_i32 s19, 0x2800
	s_mov_b32 s28, 0
	s_mov_b32 s6, s11
	s_branch .LBB5_8
.LBB5_7:                                ;   in Loop: Header=BB5_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_14
.LBB5_8:                                ; %.preheader502.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_10 Depth 2
	s_mov_b32 s7, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s6, 1
	s_mul_u64 s[24:25], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s26, s15
	s_mov_b32 s31, 0
	s_cselect_b32 s27, -1, 0
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_mov_b32 s30, 0
	s_mov_b32 s29, -1
	s_branch .LBB5_10
.LBB5_9:                                ;   in Loop: Header=BB5_10 Depth=2
	s_wait_dscnt 0xb
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v172, v172, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v179, v179, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v177, v177, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v178, v178, v57 :: v_dual_add_f32 v175, v175, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_dual_add_f32 v176, v176, v59 :: v_dual_add_f32 v173, v173, v60
	v_add_f32_e32 v174, v174, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
	v_dual_add_f32 v170, v170, v49 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_dual_mul_f32 v58, v102, v92 :: v_dual_add_f32 v171, v171, v50
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v52, v59
	v_dual_add_f32 v169, v169, v51 :: v_dual_mul_f32 v52, v96, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v55, v103, v92
	v_mul_f32_e32 v57, v98, v92
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	v_mul_f32_e32 v54, v57, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	v_dual_add_f32 v168, v168, v49 :: v_dual_add_f32 v167, v167, v51
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_add_f32_e32 v166, v166, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_dual_mul_f32 v49, v111, v90 :: v_dual_add_f32 v164, v164, v52
	v_add_f32_e32 v165, v165, v54
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v162, v162, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v163, v163, v42 :: v_dual_add_f32 v160, v160, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v161, v161, v41 :: v_dual_add_f32 v158, v158, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v159, v159, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v156, v156, v44 :: v_dual_add_f32 v157, v157, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_dual_mul_f32 v42, v107, v72 :: v_dual_add_f32 v153, v153, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v44, v43
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v33, v41, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v36, v105, v72 :: v_dual_fmac_f32 v35, v42, v43
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_dual_add_f32 v154, v154, v34 :: v_dual_fmac_f32 v33, v36, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v151, v151, v35 :: v_dual_mul_f32 v34, v41, v37
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v37, v103, v72 :: v_dual_mul_f32 v38, v96, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v98, v72 :: v_dual_add_f32 v152, v152, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_mul_f32 v38, v99, v72 :: v_dual_add_f32 v149, v149, v34
	v_fmac_f32_e32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v150, v150, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v147, v147, v33
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v145, v145, v25 :: v_dual_add_f32 v146, v146, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v143, v143, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v144, v144, v27
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v142, v142, v28
	v_mul_f32_e32 v28, v94, v75
	v_fmac_f32_e32 v25, v26, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	v_dual_add_f32 v141, v141, v25 :: v_dual_fmac_f32 v26, v28, v95
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v28, v92, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v92, v84
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v139, v139, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v25, v92, v83
	v_add_f32_e32 v138, v138, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v140, v140, v27 :: v_dual_add_f32 v137, v137, v17
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_mul_f32_e32 v19, v29, v19
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_add_f32_e32 v136, v136, v20
	v_mul_f32_e32 v20, v92, v74
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	v_add_f32_e32 v133, v133, v17
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v131, v131, v20 :: v_dual_mul_f32 v20, v90, v82
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, v21, v12 :: v_dual_mul_f32 v11, v20, v11
	v_mul_f32_e32 v20, v90, v78
	v_fmac_f32_e32 v17, v18, v59
	v_mul_f32_e32 v10, v22, v10
	v_add_f32_e32 v134, v134, v19
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v87
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_fmac_f32 v9, v18, v51
	v_dual_mul_f32 v18, v90, v85 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v90, v74
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v129, v129, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_add_f32_e32 v128, v128, v12
	v_dual_add_f32 v132, v132, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	s_barrier_signal -1
	v_add_f32_e32 v148, v148, v36
	s_xor_b32 s10, s29, -1
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	s_mov_b32 s31, 1
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s10
	v_add_f32_e32 v127, v127, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	s_mov_b32 s30, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v126, v126, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_add_f32_e32 v125, v125, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	v_fmac_f32_e32 v1, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v121, v121, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v123, v123, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v122, v122, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v120, v120, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v118, v118, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v119, v119, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v116, v116, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v117, v117, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_7
.LBB5_10:                               ;   Parent Loop BB5_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s33, s31, s7
	v_add_nc_u32_e32 v86, s28, v184
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s10, s33, 1
	s_and_b32 s28, s30, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[10:11], s[2:3]
	s_lshl_b32 s10, s31, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[22:23]
	ds_load_2addr_stride64_b64 v[74:77], v86 offset1:1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[34:35], s[34:35], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, vcc_lo, v190, s34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, s35, v191, vcc_lo
	s_add_nc_u64 s[34:35], s[24:25], s[10:11]
	s_clause 0x1
	global_load_b64 v[1:2], v[72:73], off
	global_load_b64 v[3:4], v[72:73], off offset:512
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v7, s10, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s10
	v_add_co_u32 v9, s10, s34, v69
	s_clause 0x1
	global_load_b64 v[5:6], v[72:73], off offset:1024
	global_load_b64 v[78:79], v[72:73], off offset:1536
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s35, 0, s10
	s_clause 0x1
	global_load_b64 v[112:113], v[7:8], off offset:40
	global_load_b64 v[114:115], v[9:10], off offset:40
	s_wait_loadcnt_dscnt 0x500
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[5:6], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[5:6], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[72:73], off offset:256
	global_load_b64 v[80:81], v[72:73], off offset:768
	global_load_b64 v[82:83], v[72:73], off offset:1280
	global_load_b64 v[84:85], v[72:73], off offset:1792
	ds_load_2addr_b64 v[74:77], v86 offset0:32 offset1:96
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_store_b64 v182, v[112:113] offset:4096
	ds_store_b64 v183, v[114:115] offset:4096
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_12
; %bb.11:                               ;   in Loop: Header=BB5_10 Depth=2
	s_add_co_i32 s33, s33, 1
	s_add_co_i32 s10, s31, s6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[74:75], null, s33, s14, v[70:71]
	s_mul_u64 s[34:35], s[10:11], 0x88
	s_xor_b32 s31, s31, 1
	s_mov_b32 s37, s11
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s31, 6
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_mulk_i32 s10, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_mad_co_u64_u32 v[75:76], null, s33, s13, v[75:76]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v76, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, s35, 0, s33
	v_add_co_u32 v78, s33, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s33
	v_lshlrev_b64_e32 v[74:75], 3, v[74:75]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, v185, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v186, v75, vcc_lo
	v_add_co_u32 v80, vcc_lo, v65, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v66, vcc_lo
	s_lshl_b32 s10, s31, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, vcc_lo, v80, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[112:113], v[76:77], off offset:8
	global_load_b64 v[114:115], v[78:79], off offset:8
	global_load_b32 v193, v[74:75], off
	global_load_b32 v194, v[80:81], off
.LBB5_12:                               ; %.preheader500.i
                                        ;   in Loop: Header=BB5_10 Depth=2
	v_add_co_u32 v76, vcc_lo, v72, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s9, v73, vcc_lo
	v_add_nc_u32_e32 v86, 0, v184
	s_xor_b32 s10, s28, -1
	s_and_b32 s28, s29, exec_lo
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off
	global_load_b64 v[80:81], v[76:77], off offset:512
	global_load_b64 v[82:83], v[76:77], off offset:1024
	global_load_b64 v[84:85], v[76:77], off offset:1536
	s_cselect_b32 s28, s17, 0x2400
	ds_load_2addr_stride64_b64 v[72:75], v86 offset0:8 offset1:9
	s_cselect_b32 s31, s19, 0x2c00
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off offset:256
	global_load_b64 v[80:81], v[76:77], off offset:768
	global_load_b64 v[82:83], v[76:77], off offset:1280
	global_load_b64 v[76:77], v[76:77], off offset:1792
	v_add_nc_u32_e32 v72, 0x100, v86
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:8 offset1:9
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[76:77], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[76:77], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s31, v189
	v_add_nc_u32_e32 v72, s28, v192
	s_and_not1_b32 vcc_lo, exec_lo, s10
	s_movk_i32 s28, 0x1000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_9
; %bb.13:                               ; %.preheader501.i
                                        ;   in Loop: Header=BB5_10 Depth=2
	v_lshrrev_b32_e32 v195, v188, v194
	s_and_b32 s10, s30, exec_lo
	s_cselect_b32 s10, s17, 0x2400
	s_cselect_b32 s28, s19, 0x2c00
	v_cndmask_b32_e64 v196, 0, v193, s0
	v_cvt_f32_f16_e64 v195, v195.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v197, s10, v187
	v_add_nc_u32_e32 v198, s28, v187
	s_mov_b32 s28, 0
	v_cndmask_b32_e64 v195, 0, v195, s1
	ds_store_b64 v182, v[112:113]
	ds_store_b64 v183, v[114:115]
	ds_store_b32 v197, v196
	ds_store_b32 v198, v195
	s_branch .LBB5_9
.LBB5_14:                               ; %._crit_edge564.i
	v_mul_u32_u24_e32 v1, 0x500, v181
	v_lshlrev_b32_e32 v2, 2, v155
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v5, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v3, 0x50, v155
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x280, v180, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_105
; %bb.15:                               ; %.preheader497.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v155
	v_or_b32_e32 v8, s18, v5
	v_add3_u32 v6, 0, v3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB5_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	ds_load_b32 v12, v6
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off
.LBB5_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:1280
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off
.LBB5_19:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:2560
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
.LBB5_21:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:3840
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
.LBB5_23:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:5120
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
.LBB5_25:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:6400
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
.LBB5_27:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:7680
	global_load_b32 v12, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:384
.LBB5_29:
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v180
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB5_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v6 offset:8960
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384
.LBB5_31:                               ; %.preheader495.1.i
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v170, v171 offset1:20
	ds_store_2addr_b32 v7, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB5_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s2, s20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v14, v6
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off
.LBB5_33:
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_110
.LBB5_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_111
.LBB5_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_112
.LBB5_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_113
.LBB5_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_114
.LBB5_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_41
.LBB5_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:8960
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
.LBB5_41:                               ; %.preheader495.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v162, v163 offset1:20
	ds_store_2addr_b32 v7, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB5_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	ds_load_b32 v16, v6
	global_load_b32 v15, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off
.LBB5_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_116
.LBB5_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_117
.LBB5_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_118
.LBB5_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_119
.LBB5_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_120
.LBB5_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_51
.LBB5_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384
.LBB5_51:                               ; %.preheader495.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v153, v154 offset1:20
	ds_store_2addr_b32 v7, v151, v152 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v149, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v147, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB5_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	ds_load_b32 v18, v6
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off
.LBB5_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB5_121
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB5_122
.LBB5_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB5_123
.LBB5_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB5_124
.LBB5_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB5_125
.LBB5_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB5_126
.LBB5_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB5_61
.LBB5_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:8960
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384
.LBB5_61:                               ; %.preheader496.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v145, v146 offset1:20
	ds_store_2addr_b32 v7, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB5_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64
.LBB5_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB5_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64
.LBB5_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB5_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192
.LBB5_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB5_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192
.LBB5_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB5_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320
.LBB5_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB5_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320
.LBB5_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB5_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v8, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v8, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	ds_load_b32 v18, v6 offset:7680
	global_load_b32 v8, v[16:17], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v18, v8
	global_store_b32 v[16:17], v8, off offset:448
.LBB5_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:448
.LBB5_77:                               ; %.preheader495.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v137, v138 offset1:20
	ds_store_2addr_b32 v7, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_128
.LBB5_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_129
.LBB5_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_130
.LBB5_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_131
.LBB5_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB5_132
.LBB5_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_133
.LBB5_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_86
.LBB5_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
.LBB5_86:                               ; %.preheader495.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v129, v130 offset1:20
	ds_store_2addr_b32 v7, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_135
.LBB5_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_136
.LBB5_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_137
.LBB5_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_138
.LBB5_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_139
.LBB5_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_140
.LBB5_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_95
.LBB5_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
.LBB5_95:                               ; %.preheader495.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_142
.LBB5_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_143
.LBB5_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_144
.LBB5_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_145
.LBB5_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_146
.LBB5_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_147
.LBB5_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_104
.LBB5_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	ds_load_b32 v6, v6 offset:8960
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v7, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v8, v1, vcc_lo
	global_load_b32 v7, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	global_store_b32 v[0:1], v6, off offset:448
.LBB5_104:                              ; %.loopexit.loopexit597.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB5_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB5_107
; %bb.106:                              ; %.preheader494.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_lo_u32 v0, s12, v5
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[2:3], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v3, 0, v3, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v5, v0, v155, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, -1
	s_lshl_b32 s5, s12, 8
	s_movk_i32 s4, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v6, v5, s[0:3], null offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_or_b32 s6, s5, 0x80
	s_mul_i32 s7, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s7, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v6
	buffer_store_b32 v0, v5, s[0:3], null offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s6, s5, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s6, s5, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_lshl_b32 s4, s12, 6
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v170, v171 offset1:20
	ds_store_2addr_b32 v2, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s8, s7, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s8, s7, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_lshl_b32 s6, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s6, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s8, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v162, v163 offset1:20
	ds_store_2addr_b32 v2, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s10, s8, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s10, s8, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_mul_i32 s9, s12, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s12, s10, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v153, v154 offset1:20
	ds_store_2addr_b32 v2, v151, v152 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v149, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v147, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s10, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s12, s10, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_mov_b32 s11, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	s_or_b32 s12, s5, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v145, v146 offset1:20
	ds_store_2addr_b32 v2, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_or_b32 s12, s5, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s11, s7, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s5, s4, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v137, v138 offset1:20
	ds_store_2addr_b32 v2, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s11, s7, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s11, s7, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_or_b32 s5, s8, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_or_b32 s4, s6, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v129, v130 offset1:20
	ds_store_2addr_b32 v2, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s8, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0x140
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s8, 0x140
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s5, s10, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s10, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s10, 0x140
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen
	buffer_load_b32 v2, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s9 offen
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB5_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB5_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1ELb1ELb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB5_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:1280
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_35
.LBB5_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:2560
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_36
.LBB5_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:3840
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_37
.LBB5_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:5120
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_38
.LBB5_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:6400
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB5_39
.LBB5_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:7680
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB5_40
	s_branch .LBB5_41
.LBB5_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_45
.LBB5_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_46
.LBB5_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_47
.LBB5_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_48
.LBB5_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB5_49
.LBB5_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:7680
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB5_50
	s_branch .LBB5_51
.LBB5_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB5_55
.LBB5_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB5_56
.LBB5_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB5_57
.LBB5_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB5_58
.LBB5_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB5_59
.LBB5_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:7680
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB5_60
	s_branch .LBB5_61
.LBB5_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_79
.LBB5_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_80
.LBB5_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_81
.LBB5_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_82
.LBB5_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB5_83
.LBB5_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_84
.LBB5_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v16, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v16, v10
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_85
	s_branch .LBB5_86
.LBB5_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_88
.LBB5_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:1280
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_89
.LBB5_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:2560
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_90
.LBB5_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:3840
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_91
.LBB5_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:5120
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_92
.LBB5_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:6400
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_93
.LBB5_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_94
	s_branch .LBB5_95
.LBB5_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_97
.LBB5_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:1280
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_98
.LBB5_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:2560
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_99
.LBB5_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:3840
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_100
.LBB5_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:5120
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_101
.LBB5_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:6400
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB5_102
.LBB5_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:7680
	global_load_b32 v9, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB5_103
	s_branch .LBB5_104
.Lfunc_end5:
	.size	gemm_mq4g256v2_residual_mmq_iu4_atiled_add, .Lfunc_end5-gemm_mq4g256v2_residual_mmq_iu4_atiled_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_atiled_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 199
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-gemm_mq4g256v2_residual_mmq_iu4_atiled_add)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.num_vgpr, 199
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18572
; TotalNumSgprs: 40
; NumVgprs: 199
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 199
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_atiled_set ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_atiled_set
	.globl	gemm_mq4g256v2_residual_mmq_iu4_atiled_set
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_atiled_set,@function
gemm_mq4g256v2_residual_mmq_iu4_atiled_set: ; @gemm_mq4g256v2_residual_mmq_iu4_atiled_set
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB6_108
; %bb.1:                                ; %.preheader501.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_add_co_i32 s2, s14, 0x7f
	s_ashr_i32 s1, s13, 31
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s10, s1, 24
	s_lshr_b32 s0, s3, 25
	s_add_co_i32 s10, s13, s10
	s_add_co_i32 s2, s2, s0
	s_lshr_b32 s0, s1, 26
	s_ashr_i32 s2, s2, 7
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s0, s0, 6
	s_lshl_b64 s[8:9], s[2:3], 12
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v3, 1, v0
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	s_ashr_i32 s15, s10, 8
	s_add_co_i32 s17, s12, -1
	s_mul_i32 s10, s15, 0x88
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB6_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s18, v4
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v3
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v1
	s_cbranch_execz .LBB6_4
; %bb.3:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 3, v[1:2]
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB6_4:                                ; %.preheader496.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v1, s16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v6, s17, v1
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mad_co_i64_i32 v[6:7], null, s10, v6, s[4:5]
	global_load_b32 v6, v[6:7], off
	v_lshlrev_b32_e32 v7, 4, v3
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v6, v7, v6
	v_lshlrev_b32_e32 v7, 3, v4
	v_cvt_f32_f16_e32 v6, v6.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v5, 0, v7, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v6, vcc_lo
	ds_store_2addr_stride64_b32 v5, v2, v1 offset0:32 offset1:40
.LBB6_5:                                ; %Flow821
	s_or_b32 exec_lo, exec_lo, s11
	v_lshrrev_b32_e32 v7, 2, v0
	v_dual_mov_b32 v116, 0 :: v_dual_and_b32 v1, 3, v0
	v_dual_mov_b32 v148, 0 :: v_dual_lshlrev_b32 v9, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v2, s16, v7
	v_dual_mov_b32 v120, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v168, 0 :: v_dual_and_b32 v9, 16, v9
	v_dual_mov_b32 v118, 0 :: v_dual_add_nc_u32 v5, 64, v2
	v_min_i32_e32 v2, s17, v2
	v_lshrrev_b32_e32 v181, 5, v0
	v_bfe_u32 v8, v0, 1, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v5, s17, v5
	v_and_or_b32 v7, v7, 15, v9
	v_mad_co_u64_u32 v[68:69], null, s10, v2, v[1:2]
	v_or_b32_e32 v9, 8, v181
	v_and_or_b32 v10, v181, 6, v8
	v_mad_co_u64_u32 v[69:70], null, s10, v5, v[1:2]
	v_dual_mov_b32 v142, 0 :: v_dual_lshlrev_b32 v7, 3, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v8, v9, 14, v8
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v145, 15, v0
	v_mov_b32_e32 v122, 0
	s_clause 0x1
	global_load_b64 v[1:2], v68, s[4:5] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	v_lshl_or_b32 v9, v10, 8, v7
	v_lshl_or_b32 v7, v8, 8, v7
	v_mov_b32_e32 v137, 0
	v_bfe_u32 v180, v0, 4, 1
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_add_nc_u32_e32 v182, 0, v9
	v_add_nc_u32_e32 v183, 0, v7
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s11, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x1
	ds_store_b64 v182, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b64 v183, v[5:6]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB6_14
; %bb.6:                                ; %.preheader495.lr.ph.i
	v_dual_mov_b32 v172, 0 :: v_dual_and_b32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 6, v0
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v7, s16, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v194, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v193, 0 :: v_dual_add_nc_u32 v6, s18, v4
	v_dual_mov_b32 v175, 0 :: v_dual_lshlrev_b32 v10, 2, v3
	v_lshl_or_b32 v9, v5, 11, v1
	v_lshl_or_b32 v184, v2, 10, v1
	v_min_i32_e32 v1, s17, v7
	v_dual_mov_b32 v173, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	s_add_co_i32 s13, s14, -1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s10, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v70, s13, v6
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v2, 6, v180
	v_add3_u32 v187, 0, v4, v10
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v4, 3, v145
	v_mad_co_u64_u32 v[66:67], null, s10, v1, v[66:67]
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v1, 9, v5
	v_add_co_u32 v185, s0, s0, v10
	v_add_co_u32 v190, s6, s6, v9
	v_ashrrev_i32_e32 v71, 31, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v186, null, s1, 0, s0
	v_cmp_gt_i32_e64 s0, s14, v6
	v_cmp_gt_i32_e64 s1, s12, v7
	v_dual_mov_b32 v143, 0 :: v_dual_lshlrev_b32 v188, 4, v3
	v_add3_u32 v189, 0, v8, v2
	v_add_co_ci_u32_e64 v191, null, s7, 0, s6
	v_add3_u32 v192, 0, v1, v4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	s_mov_b32 s22, ttmp7
	s_mov_b32 s23, s11
	s_ashr_i32 s13, s14, 31
	s_movk_i32 s17, 0x2000
	s_movk_i32 s19, 0x2800
	s_mov_b32 s28, 0
	s_mov_b32 s6, s11
	s_branch .LBB6_8
.LBB6_7:                                ;   in Loop: Header=BB6_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_14
.LBB6_8:                                ; %.preheader495.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_10 Depth 2
	s_mov_b32 s7, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s6, 1
	s_mul_u64 s[24:25], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s26, s15
	s_mov_b32 s31, 0
	s_cselect_b32 s27, -1, 0
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_mov_b32 s30, 0
	s_mov_b32 s29, -1
	s_branch .LBB6_10
.LBB6_9:                                ;   in Loop: Header=BB6_10 Depth=2
	s_wait_dscnt 0xb
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v172, v172, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v179, v179, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v177, v177, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v178, v178, v57 :: v_dual_add_f32 v175, v175, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_dual_add_f32 v176, v176, v59 :: v_dual_add_f32 v173, v173, v60
	v_add_f32_e32 v174, v174, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
	v_dual_add_f32 v170, v170, v49 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_dual_mul_f32 v58, v102, v92 :: v_dual_add_f32 v171, v171, v50
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v52, v59
	v_dual_add_f32 v169, v169, v51 :: v_dual_mul_f32 v52, v96, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v55, v103, v92
	v_mul_f32_e32 v57, v98, v92
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	v_mul_f32_e32 v54, v57, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	v_dual_add_f32 v168, v168, v49 :: v_dual_add_f32 v167, v167, v51
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_add_f32_e32 v166, v166, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_dual_mul_f32 v49, v111, v90 :: v_dual_add_f32 v164, v164, v52
	v_add_f32_e32 v165, v165, v54
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v162, v162, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v163, v163, v42 :: v_dual_add_f32 v160, v160, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v161, v161, v41 :: v_dual_add_f32 v158, v158, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v159, v159, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v156, v156, v44 :: v_dual_add_f32 v157, v157, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v42, v107, v72
	v_cvt_f32_i32_e32 v37, v37
	v_fmac_f32_e32 v34, v44, v43
	v_dual_add_f32 v154, v154, v33 :: v_dual_mul_f32 v33, v41, v36
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v36, v105, v72
	v_fmac_f32_e32 v35, v42, v43
	v_mul_f32_e32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_add_f32_e32 v155, v155, v34
	v_fmac_f32_e32 v33, v36, v43
	v_add_f32_e32 v152, v152, v35
	v_dual_mul_f32 v34, v41, v37 :: v_dual_mul_f32 v37, v103, v72
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v41, v98, v72 :: v_dual_mul_f32 v38, v96, v72
	v_add_f32_e32 v153, v153, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	v_mul_f32_e32 v38, v99, v72
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_add_f32 v150, v150, v34 :: v_dual_fmac_f32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_mul_f32 v25, v39, v25
	v_dual_mul_f32 v26, v40, v26 :: v_dual_mul_f32 v37, v94, v87
	v_dual_mul_f32 v34, v94, v89 :: v_dual_add_f32 v151, v151, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v148, v148, v33 :: v_dual_add_f32 v149, v149, v36
	v_dual_fmac_f32 v26, v37, v95 :: v_dual_fmac_f32 v25, v34, v95
	s_wait_dscnt 0x4
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v146, v146, v25 :: v_dual_add_f32 v147, v147, v26
	v_mul_f32_e32 v26, v94, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v143, v143, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v144, v144, v27
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v142, v142, v28
	v_mul_f32_e32 v28, v94, v75
	v_fmac_f32_e32 v25, v26, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	v_cvt_f32_i32_e32 v18, v18
	v_dual_add_f32 v141, v141, v25 :: v_dual_fmac_f32 v26, v28, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v28, v92, v87
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v92, v84
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v139, v139, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v25, v92, v83
	v_add_f32_e32 v138, v138, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v140, v140, v27 :: v_dual_add_f32 v137, v137, v17
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_mul_f32_e32 v19, v29, v19
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_add_f32_e32 v136, v136, v20
	v_mul_f32_e32 v20, v92, v74
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	v_add_f32_e32 v133, v133, v17
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v131, v131, v20 :: v_dual_mul_f32 v20, v90, v82
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, v21, v12 :: v_dual_mul_f32 v11, v20, v11
	v_mul_f32_e32 v20, v90, v78
	v_fmac_f32_e32 v17, v18, v59
	v_mul_f32_e32 v10, v22, v10
	v_add_f32_e32 v134, v134, v19
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v87
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_fmac_f32 v9, v18, v51
	v_dual_mul_f32 v18, v90, v85 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v90, v74
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v129, v129, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_add_f32_e32 v128, v128, v12
	v_dual_add_f32 v132, v132, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	s_barrier_signal -1
	s_xor_b32 s10, s29, -1
	s_mov_b32 s31, 1
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s10
	s_mov_b32 s30, -1
	v_add_f32_e32 v127, v127, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v126, v126, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_add_f32_e32 v125, v125, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	v_fmac_f32_e32 v1, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v121, v121, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v123, v123, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v122, v122, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v120, v120, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v118, v118, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v119, v119, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v116, v116, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v117, v117, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_7
.LBB6_10:                               ;   Parent Loop BB6_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s33, s31, s7
	v_add_nc_u32_e32 v86, s28, v184
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s10, s33, 1
	s_and_b32 s28, s30, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[10:11], s[2:3]
	s_lshl_b32 s10, s31, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[22:23]
	ds_load_2addr_stride64_b64 v[74:77], v86 offset1:1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[34:35], s[34:35], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, vcc_lo, v190, s34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, s35, v191, vcc_lo
	s_add_nc_u64 s[34:35], s[24:25], s[10:11]
	s_clause 0x1
	global_load_b64 v[1:2], v[72:73], off
	global_load_b64 v[3:4], v[72:73], off offset:512
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v7, s10, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s10
	v_add_co_u32 v9, s10, s34, v69
	s_clause 0x1
	global_load_b64 v[5:6], v[72:73], off offset:1024
	global_load_b64 v[78:79], v[72:73], off offset:1536
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s35, 0, s10
	s_clause 0x1
	global_load_b64 v[112:113], v[7:8], off offset:40
	global_load_b64 v[114:115], v[9:10], off offset:40
	s_wait_loadcnt_dscnt 0x500
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[5:6], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[5:6], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[72:73], off offset:256
	global_load_b64 v[80:81], v[72:73], off offset:768
	global_load_b64 v[82:83], v[72:73], off offset:1280
	global_load_b64 v[84:85], v[72:73], off offset:1792
	ds_load_2addr_b64 v[74:77], v86 offset0:32 offset1:96
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_store_b64 v182, v[112:113] offset:4096
	ds_store_b64 v183, v[114:115] offset:4096
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_12
; %bb.11:                               ;   in Loop: Header=BB6_10 Depth=2
	s_add_co_i32 s33, s33, 1
	s_add_co_i32 s10, s31, s6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[74:75], null, s33, s14, v[70:71]
	s_mul_u64 s[34:35], s[10:11], 0x88
	s_xor_b32 s31, s31, 1
	s_mov_b32 s37, s11
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s31, 6
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_mulk_i32 s10, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_mad_co_u64_u32 v[75:76], null, s33, s13, v[75:76]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v76, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, s35, 0, s33
	v_add_co_u32 v78, s33, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s33
	v_lshlrev_b64_e32 v[74:75], 3, v[74:75]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, v185, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v186, v75, vcc_lo
	v_add_co_u32 v80, vcc_lo, v65, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v66, vcc_lo
	s_lshl_b32 s10, s31, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, vcc_lo, v80, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[112:113], v[76:77], off offset:8
	global_load_b64 v[114:115], v[78:79], off offset:8
	global_load_b32 v193, v[74:75], off
	global_load_b32 v194, v[80:81], off
.LBB6_12:                               ; %.preheader493.i
                                        ;   in Loop: Header=BB6_10 Depth=2
	v_add_co_u32 v76, vcc_lo, v72, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s9, v73, vcc_lo
	v_add_nc_u32_e32 v86, 0, v184
	s_xor_b32 s10, s28, -1
	s_and_b32 s28, s29, exec_lo
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off
	global_load_b64 v[80:81], v[76:77], off offset:512
	global_load_b64 v[82:83], v[76:77], off offset:1024
	global_load_b64 v[84:85], v[76:77], off offset:1536
	s_cselect_b32 s28, s17, 0x2400
	ds_load_2addr_stride64_b64 v[72:75], v86 offset0:8 offset1:9
	s_cselect_b32 s31, s19, 0x2c00
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off offset:256
	global_load_b64 v[80:81], v[76:77], off offset:768
	global_load_b64 v[82:83], v[76:77], off offset:1280
	global_load_b64 v[76:77], v[76:77], off offset:1792
	v_add_nc_u32_e32 v72, 0x100, v86
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:8 offset1:9
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[76:77], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[76:77], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s31, v189
	v_add_nc_u32_e32 v72, s28, v192
	s_and_not1_b32 vcc_lo, exec_lo, s10
	s_movk_i32 s28, 0x1000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_9
; %bb.13:                               ; %.preheader494.i
                                        ;   in Loop: Header=BB6_10 Depth=2
	v_lshrrev_b32_e32 v195, v188, v194
	s_and_b32 s10, s30, exec_lo
	s_cselect_b32 s10, s17, 0x2400
	s_cselect_b32 s28, s19, 0x2c00
	v_cndmask_b32_e64 v196, 0, v193, s0
	v_cvt_f32_f16_e64 v195, v195.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v197, s10, v187
	v_add_nc_u32_e32 v198, s28, v187
	s_mov_b32 s28, 0
	v_cndmask_b32_e64 v195, 0, v195, s1
	ds_store_b64 v182, v[112:113]
	ds_store_b64 v183, v[114:115]
	ds_store_b32 v197, v196
	ds_store_b32 v198, v195
	s_branch .LBB6_9
.LBB6_14:                               ; %._crit_edge557.i
	v_mul_u32_u24_e32 v1, 0x500, v181
	v_lshlrev_b32_e32 v2, 2, v145
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v3, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v4, 0x50, v145
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v3
	v_mad_u32_u24 v2, 0x280, v180, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB6_105
; %bb.15:                               ; %.preheader490.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v145
	v_or_b32_e32 v8, s18, v3
	v_add3_u32 v6, 0, v4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB6_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v6
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB6_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
.LBB6_19:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB6_21:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB6_23:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB6_25:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB6_27:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB6_29:
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v180
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB6_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	ds_load_b32 v15, v6 offset:8960
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:384
.LBB6_31:                               ; %.preheader488.1.i
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v170, v171 offset1:20
	ds_store_2addr_b32 v7, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB6_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v6
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v11, s2, s20, v11
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB6_33:
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_110
.LBB6_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_111
.LBB6_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_112
.LBB6_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_113
.LBB6_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_114
.LBB6_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_41
.LBB6_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
.LBB6_41:                               ; %.preheader488.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v162, v163 offset1:20
	ds_store_2addr_b32 v7, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB6_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	ds_load_b32 v17, v6
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v17, off
.LBB6_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_116
.LBB6_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_117
.LBB6_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_118
.LBB6_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_119
.LBB6_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_120
.LBB6_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_51
.LBB6_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
.LBB6_51:                               ; %.preheader488.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v154, v155 offset1:20
	ds_store_2addr_b32 v7, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v150, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v148, v149 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB6_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	ds_load_b32 v19, v6
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v19, off
.LBB6_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB6_121
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB6_122
.LBB6_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB6_123
.LBB6_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB6_124
.LBB6_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB6_125
.LBB6_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB6_126
.LBB6_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB6_61
.LBB6_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:8960
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384
.LBB6_61:                               ; %.preheader489.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v146, v147 offset1:20
	ds_store_2addr_b32 v7, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB6_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64
.LBB6_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB6_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64
.LBB6_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB6_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192
.LBB6_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB6_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192
.LBB6_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB6_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320
.LBB6_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB6_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320
.LBB6_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB6_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v8, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v8, off offset:448
.LBB6_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:448
.LBB6_77:                               ; %.preheader488.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v137, v138 offset1:20
	ds_store_2addr_b32 v7, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_128
.LBB6_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_129
.LBB6_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_130
.LBB6_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_131
.LBB6_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB6_132
.LBB6_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_133
.LBB6_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_86
.LBB6_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:448
.LBB6_86:                               ; %.preheader488.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v129, v130 offset1:20
	ds_store_2addr_b32 v7, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_135
.LBB6_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_136
.LBB6_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_137
.LBB6_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_138
.LBB6_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_139
.LBB6_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_140
.LBB6_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_95
.LBB6_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v12, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448
.LBB6_95:                               ; %.preheader488.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_142
.LBB6_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_143
.LBB6_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_144
.LBB6_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_145
.LBB6_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_146
.LBB6_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_147
.LBB6_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_104
.LBB6_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v9, v6 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v7, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v9, off offset:448
.LBB6_104:                              ; %.loopexit.loopexit590.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB6_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB6_107
; %bb.106:                              ; %.preheader487.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add3_u32 v10, 0, v4, v5
	v_mul_lo_u32 v3, s12, v3
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[4:5], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v11, v3, v145, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_mov_b32 s2, -1
	s_lshl_b32 s6, s12, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_movk_i32 s7, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[8:9], v10 offset0:30 offset1:35
	s_movk_i32 s8, 0x100
	s_movk_i32 s9, 0x180
	s_or_b32 s10, s6, 0x80
	s_add_co_i32 s4, s6, 0x100
	s_add_co_i32 s5, s6, 0x180
	s_add_co_i32 s17, s6, 0x140
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], null offen
	buffer_store_b32 v1, v11, s[0:3], s6 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s7 offen
	buffer_store_b32 v5, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s8 offen
	buffer_store_b32 v7, v11, s[0:3], s4 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v8, v11, s[0:3], s9 offen
	buffer_store_b32 v9, v11, s[0:3], s5 offen
	s_lshl_b32 s4, s12, 6
	s_mul_i32 s5, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 0x80
	s_add_co_i32 s8, s5, 0x80
	s_add_co_i32 s9, s4, 0x100
	s_add_co_i32 s10, s5, 0x100
	s_add_co_i32 s11, s4, 0x180
	s_add_co_i32 s13, s5, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v170, v171 offset1:20
	ds_store_2addr_b32 v2, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	buffer_store_b32 v4, v11, s[0:3], s8 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s9 offen
	buffer_store_b32 v6, v11, s[0:3], s10 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s11 offen
	buffer_store_b32 v8, v11, s[0:3], s13 offen
	s_lshl_b32 s7, s12, 7
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s7, 0x80
	s_add_co_i32 s10, s8, 0x80
	s_add_co_i32 s11, s7, 0x100
	s_add_co_i32 s13, s8, 0x100
	s_add_co_i32 s14, s7, 0x180
	s_add_co_i32 s15, s8, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v162, v163 offset1:20
	ds_store_2addr_b32 v2, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s7 offen
	buffer_store_b32 v1, v11, s[0:3], s8 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s9 offen
	buffer_store_b32 v4, v11, s[0:3], s10 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	buffer_store_b32 v6, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s14 offen
	buffer_store_b32 v8, v11, s[0:3], s15 offen
	s_mul_i32 s9, s12, 0xc0
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_add_co_i32 s12, s10, 0x80
	s_add_co_i32 s13, s9, 0x100
	s_add_co_i32 s14, s10, 0x100
	s_add_co_i32 s15, s9, 0x180
	s_add_co_i32 s16, s10, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v154, v155 offset1:20
	ds_store_2addr_b32 v2, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v150, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v148, v149 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s9 offen
	buffer_store_b32 v1, v11, s[0:3], s10 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s11 offen
	buffer_store_b32 v4, v11, s[0:3], s12 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s13 offen
	buffer_store_b32 v6, v11, s[0:3], s14 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s15 offen
	buffer_store_b32 v8, v11, s[0:3], s16 offen
	s_mov_b32 s14, 64
	s_or_b32 s15, s6, 64
	s_movk_i32 s11, 0x140
	s_movk_i32 s12, 0xc0
	s_movk_i32 s13, 0x1c0
	s_or_b32 s16, s6, 0xc0
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v146, v147 offset1:20
	ds_store_2addr_b32 v2, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s14 offen
	buffer_store_b32 v1, v11, s[0:3], s15 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen
	buffer_store_b32 v4, v11, s[0:3], s16 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	buffer_store_b32 v6, v11, s[0:3], s17 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s13 offen
	buffer_store_b32 v8, v11, s[0:3], s6 offen
	s_add_co_i32 s6, s4, 64
	s_add_co_i32 s11, s5, 64
	s_add_co_i32 s12, s4, 0xc0
	s_add_co_i32 s13, s5, 0xc0
	s_add_co_i32 s14, s4, 0x140
	s_add_co_i32 s15, s5, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v137, v138 offset1:20
	ds_store_2addr_b32 v2, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s6 offen
	buffer_store_b32 v1, v11, s[0:3], s11 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen
	buffer_store_b32 v4, v11, s[0:3], s13 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s14 offen
	buffer_store_b32 v6, v11, s[0:3], s15 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s4 offen
	buffer_store_b32 v8, v11, s[0:3], s5 offen
	s_or_b32 s4, s7, 64
	s_or_b32 s5, s8, 64
	s_add_co_i32 s6, s7, 0xc0
	s_add_co_i32 s11, s8, 0xc0
	s_add_co_i32 s12, s7, 0x140
	s_add_co_i32 s13, s8, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v129, v130 offset1:20
	ds_store_2addr_b32 v2, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s6 offen
	buffer_store_b32 v4, v11, s[0:3], s11 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s12 offen
	buffer_store_b32 v6, v11, s[0:3], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s7 offen
	buffer_store_b32 v8, v11, s[0:3], s8 offen
	s_add_co_i32 s4, s9, 64
	s_add_co_i32 s5, s10, 64
	s_add_co_i32 s6, s9, 0xc0
	s_add_co_i32 s7, s10, 0xc0
	s_add_co_i32 s8, s9, 0x140
	s_add_co_i32 s11, s10, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen
	buffer_store_b32 v1, v11, s[0:3], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s6 offen
	buffer_store_b32 v3, v11, s[0:3], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s8 offen
	buffer_store_b32 v5, v11, s[0:3], s11 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s9 offen
	buffer_store_b32 v7, v11, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB6_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB6_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0ELb1ELb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB6_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_35
.LBB6_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_36
.LBB6_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_37
.LBB6_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_38
.LBB6_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB6_39
.LBB6_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB6_40
	s_branch .LBB6_41
.LBB6_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_45
.LBB6_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_46
.LBB6_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_47
.LBB6_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_48
.LBB6_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB6_49
.LBB6_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB6_50
	s_branch .LBB6_51
.LBB6_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB6_55
.LBB6_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB6_56
.LBB6_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB6_57
.LBB6_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB6_58
.LBB6_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB6_59
.LBB6_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB6_60
	s_branch .LBB6_61
.LBB6_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_79
.LBB6_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_80
.LBB6_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_81
.LBB6_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_82
.LBB6_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB6_83
.LBB6_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_84
.LBB6_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v10, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v10, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_85
	s_branch .LBB6_86
.LBB6_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_88
.LBB6_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_89
.LBB6_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_90
.LBB6_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_91
.LBB6_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_92
.LBB6_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_93
.LBB6_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v12, v6 offset:7680
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_94
	s_branch .LBB6_95
.LBB6_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_97
.LBB6_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:1280
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_98
.LBB6_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:2560
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_99
.LBB6_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:3840
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_100
.LBB6_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:5120
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_101
.LBB6_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:6400
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB6_102
.LBB6_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:7680
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB6_103
	s_branch .LBB6_104
.Lfunc_end6:
	.size	gemm_mq4g256v2_residual_mmq_iu4_atiled_set, .Lfunc_end6-gemm_mq4g256v2_residual_mmq_iu4_atiled_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_atiled_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 199
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-gemm_mq4g256v2_residual_mmq_iu4_atiled_set)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.num_vgpr, 199
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16280
; TotalNumSgprs: 40
; NumVgprs: 199
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 199
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
	.globl	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add,@function
gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add: ; @gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB7_108
; %bb.1:                                ; %.preheader508.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_add_co_i32 s2, s14, 0x7f
	s_ashr_i32 s1, s13, 31
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s10, s1, 24
	s_lshr_b32 s0, s3, 25
	s_add_co_i32 s10, s13, s10
	s_add_co_i32 s2, s2, s0
	s_lshr_b32 s0, s1, 26
	s_ashr_i32 s2, s2, 7
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s0, s0, 6
	s_lshl_b64 s[8:9], s[2:3], 12
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v3, 1, v0
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	s_ashr_i32 s15, s10, 8
	s_add_co_i32 s17, s12, -1
	s_mul_i32 s10, s15, 0x88
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB7_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s18, v4
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v3
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v1
	s_cbranch_execz .LBB7_4
; %bb.3:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 3, v[1:2]
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB7_4:                                ; %.preheader503.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v1, s16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v6, s17, v1
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mad_co_i64_i32 v[6:7], null, s10, v6, s[4:5]
	global_load_b32 v6, v[6:7], off
	v_lshlrev_b32_e32 v7, 4, v3
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v6, v7, v6
	v_lshlrev_b32_e32 v7, 3, v4
	v_cvt_f32_f16_e32 v6, v6.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v5, 0, v7, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v6, vcc_lo
	ds_store_2addr_stride64_b32 v5, v2, v1 offset0:32 offset1:40
.LBB7_5:                                ; %Flow821
	s_or_b32 exec_lo, exec_lo, s11
	v_lshrrev_b32_e32 v7, 2, v0
	v_dual_mov_b32 v116, 0 :: v_dual_and_b32 v1, 3, v0
	v_dual_mov_b32 v148, 0 :: v_dual_lshlrev_b32 v9, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v2, s16, v7
	v_dual_mov_b32 v120, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v168, 0 :: v_dual_and_b32 v9, 16, v9
	v_dual_mov_b32 v118, 0 :: v_dual_add_nc_u32 v5, 64, v2
	v_min_i32_e32 v2, s17, v2
	v_lshrrev_b32_e32 v181, 5, v0
	v_bfe_u32 v8, v0, 1, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v5, s17, v5
	v_and_or_b32 v7, v7, 15, v9
	v_mad_co_u64_u32 v[68:69], null, s10, v2, v[1:2]
	v_or_b32_e32 v9, 8, v181
	v_and_or_b32 v10, v181, 6, v8
	v_mad_co_u64_u32 v[69:70], null, s10, v5, v[1:2]
	v_dual_mov_b32 v142, 0 :: v_dual_lshlrev_b32 v7, 3, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v8, v9, 14, v8
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v155, 15, v0
	v_mov_b32_e32 v122, 0
	s_clause 0x1
	global_load_b64 v[1:2], v68, s[4:5] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	v_lshl_or_b32 v9, v10, 8, v7
	v_lshl_or_b32 v7, v8, 8, v7
	v_mov_b32_e32 v137, 0
	v_bfe_u32 v180, v0, 4, 1
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_add_nc_u32_e32 v182, 0, v9
	v_add_nc_u32_e32 v183, 0, v7
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s11, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x1
	ds_store_b64 v182, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b64 v183, v[5:6]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB7_14
; %bb.6:                                ; %.preheader502.lr.ph.i
	v_dual_mov_b32 v172, 0 :: v_dual_and_b32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 6, v0
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v7, s16, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v194, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v193, 0 :: v_dual_add_nc_u32 v6, s18, v4
	v_dual_mov_b32 v175, 0 :: v_dual_lshlrev_b32 v10, 2, v3
	v_lshl_or_b32 v9, v5, 11, v1
	v_lshl_or_b32 v184, v2, 10, v1
	v_min_i32_e32 v1, s17, v7
	v_dual_mov_b32 v173, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	s_add_co_i32 s13, s14, -1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s10, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v70, s13, v6
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v2, 6, v180
	v_add3_u32 v187, 0, v4, v10
	v_dual_mov_b32 v145, 0 :: v_dual_lshlrev_b32 v4, 3, v155
	v_mad_co_u64_u32 v[66:67], null, s10, v1, v[66:67]
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v1, 9, v5
	v_add_co_u32 v185, s0, s0, v10
	v_add_co_u32 v190, s6, s6, v9
	v_ashrrev_i32_e32 v71, 31, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v186, null, s1, 0, s0
	v_cmp_gt_i32_e64 s0, s14, v6
	v_cmp_gt_i32_e64 s1, s12, v7
	v_dual_mov_b32 v143, 0 :: v_dual_lshlrev_b32 v188, 4, v3
	v_add3_u32 v189, 0, v8, v2
	v_add_co_ci_u32_e64 v191, null, s7, 0, s6
	v_add3_u32 v192, 0, v1, v4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	s_mov_b32 s22, ttmp7
	s_mov_b32 s23, s11
	s_ashr_i32 s13, s14, 31
	s_movk_i32 s17, 0x2000
	s_movk_i32 s19, 0x2800
	s_mov_b32 s28, 0
	s_mov_b32 s6, s11
	s_branch .LBB7_8
.LBB7_7:                                ;   in Loop: Header=BB7_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB7_14
.LBB7_8:                                ; %.preheader502.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_10 Depth 2
	s_mov_b32 s7, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s6, 1
	s_mul_u64 s[24:25], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s26, s15
	s_mov_b32 s31, 0
	s_cselect_b32 s27, -1, 0
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_mov_b32 s30, 0
	s_mov_b32 s29, -1
	s_branch .LBB7_10
.LBB7_9:                                ;   in Loop: Header=BB7_10 Depth=2
	s_wait_dscnt 0xb
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v172, v172, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v179, v179, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v177, v177, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v178, v178, v57 :: v_dual_add_f32 v175, v175, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_dual_add_f32 v176, v176, v59 :: v_dual_add_f32 v173, v173, v60
	v_add_f32_e32 v174, v174, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
	v_dual_add_f32 v170, v170, v49 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_dual_mul_f32 v58, v102, v92 :: v_dual_add_f32 v171, v171, v50
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v52, v59
	v_dual_add_f32 v169, v169, v51 :: v_dual_mul_f32 v52, v96, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v55, v103, v92
	v_mul_f32_e32 v57, v98, v92
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	v_mul_f32_e32 v54, v57, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	v_dual_add_f32 v168, v168, v49 :: v_dual_add_f32 v167, v167, v51
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_add_f32_e32 v166, v166, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_dual_mul_f32 v49, v111, v90 :: v_dual_add_f32 v164, v164, v52
	v_add_f32_e32 v165, v165, v54
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v162, v162, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v163, v163, v42 :: v_dual_add_f32 v160, v160, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v161, v161, v41 :: v_dual_add_f32 v158, v158, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v159, v159, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v156, v156, v44 :: v_dual_add_f32 v157, v157, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_dual_mul_f32 v42, v107, v72 :: v_dual_add_f32 v153, v153, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v34, v44, v43
	v_cvt_f32_i32_e32 v37, v37
	v_mul_f32_e32 v33, v41, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v36, v105, v72 :: v_dual_fmac_f32 v35, v42, v43
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_dual_add_f32 v154, v154, v34 :: v_dual_fmac_f32 v33, v36, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v151, v151, v35 :: v_dual_mul_f32 v34, v41, v37
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v37, v103, v72 :: v_dual_mul_f32 v38, v96, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v98, v72 :: v_dual_add_f32 v152, v152, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_mul_f32 v38, v99, v72 :: v_dual_add_f32 v149, v149, v34
	v_fmac_f32_e32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v150, v150, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v147, v147, v33
	s_wait_dscnt 0x5
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	s_wait_dscnt 0x4
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v145, v145, v25 :: v_dual_add_f32 v146, v146, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v143, v143, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v144, v144, v27
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v142, v142, v28
	v_mul_f32_e32 v28, v94, v75
	v_fmac_f32_e32 v25, v26, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	v_dual_add_f32 v141, v141, v25 :: v_dual_fmac_f32 v26, v28, v95
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v28, v92, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v92, v84
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v139, v139, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v25, v92, v83
	v_add_f32_e32 v138, v138, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v140, v140, v27 :: v_dual_add_f32 v137, v137, v17
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_mul_f32_e32 v19, v29, v19
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_add_f32_e32 v136, v136, v20
	v_mul_f32_e32 v20, v92, v74
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	v_add_f32_e32 v133, v133, v17
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v131, v131, v20 :: v_dual_mul_f32 v20, v90, v82
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, v21, v12 :: v_dual_mul_f32 v11, v20, v11
	v_mul_f32_e32 v20, v90, v78
	v_fmac_f32_e32 v17, v18, v59
	v_mul_f32_e32 v10, v22, v10
	v_add_f32_e32 v134, v134, v19
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v87
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_fmac_f32 v9, v18, v51
	v_dual_mul_f32 v18, v90, v85 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v90, v74
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v129, v129, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_add_f32_e32 v128, v128, v12
	v_dual_add_f32 v132, v132, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	s_barrier_signal -1
	v_add_f32_e32 v148, v148, v36
	s_xor_b32 s10, s29, -1
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	s_mov_b32 s31, 1
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s10
	v_add_f32_e32 v127, v127, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	s_mov_b32 s30, -1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v126, v126, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_add_f32_e32 v125, v125, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	v_fmac_f32_e32 v1, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v121, v121, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v123, v123, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v122, v122, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v120, v120, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v118, v118, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v119, v119, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v116, v116, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v117, v117, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB7_7
.LBB7_10:                               ;   Parent Loop BB7_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s33, s31, s7
	v_add_nc_u32_e32 v86, s28, v184
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s10, s33, 1
	s_and_b32 s28, s30, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[10:11], s[2:3]
	s_lshl_b32 s10, s31, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[22:23]
	ds_load_2addr_stride64_b64 v[74:77], v86 offset1:1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[34:35], s[34:35], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, vcc_lo, v190, s34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, s35, v191, vcc_lo
	s_add_nc_u64 s[34:35], s[24:25], s[10:11]
	s_clause 0x1
	global_load_b64 v[1:2], v[72:73], off
	global_load_b64 v[3:4], v[72:73], off offset:512
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v7, s10, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s10
	v_add_co_u32 v9, s10, s34, v69
	s_clause 0x1
	global_load_b64 v[5:6], v[72:73], off offset:1024
	global_load_b64 v[78:79], v[72:73], off offset:1536
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s35, 0, s10
	s_clause 0x1
	global_load_b64 v[112:113], v[7:8], off offset:40
	global_load_b64 v[114:115], v[9:10], off offset:40
	s_wait_loadcnt_dscnt 0x500
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[5:6], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[5:6], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[72:73], off offset:256
	global_load_b64 v[80:81], v[72:73], off offset:768
	global_load_b64 v[82:83], v[72:73], off offset:1280
	global_load_b64 v[84:85], v[72:73], off offset:1792
	ds_load_2addr_b64 v[74:77], v86 offset0:32 offset1:96
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_store_b64 v182, v[112:113] offset:4096
	ds_store_b64 v183, v[114:115] offset:4096
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB7_12
; %bb.11:                               ;   in Loop: Header=BB7_10 Depth=2
	s_add_co_i32 s33, s33, 1
	s_add_co_i32 s10, s31, s6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[74:75], null, s33, s14, v[70:71]
	s_mul_u64 s[34:35], s[10:11], 0x88
	s_xor_b32 s31, s31, 1
	s_mov_b32 s37, s11
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s31, 6
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_mulk_i32 s10, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_mad_co_u64_u32 v[75:76], null, s33, s13, v[75:76]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v76, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, s35, 0, s33
	v_add_co_u32 v78, s33, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s33
	v_lshlrev_b64_e32 v[74:75], 3, v[74:75]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, v185, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v186, v75, vcc_lo
	v_add_co_u32 v80, vcc_lo, v65, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v66, vcc_lo
	s_lshl_b32 s10, s31, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, vcc_lo, v80, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[112:113], v[76:77], off offset:8
	global_load_b64 v[114:115], v[78:79], off offset:8
	global_load_b32 v193, v[74:75], off
	global_load_b32 v194, v[80:81], off
.LBB7_12:                               ; %.preheader500.i
                                        ;   in Loop: Header=BB7_10 Depth=2
	v_add_co_u32 v76, vcc_lo, v72, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s9, v73, vcc_lo
	v_add_nc_u32_e32 v86, 0, v184
	s_xor_b32 s10, s28, -1
	s_and_b32 s28, s29, exec_lo
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off
	global_load_b64 v[80:81], v[76:77], off offset:512
	global_load_b64 v[82:83], v[76:77], off offset:1024
	global_load_b64 v[84:85], v[76:77], off offset:1536
	s_cselect_b32 s28, s17, 0x2400
	ds_load_2addr_stride64_b64 v[72:75], v86 offset0:8 offset1:9
	s_cselect_b32 s31, s19, 0x2c00
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off offset:256
	global_load_b64 v[80:81], v[76:77], off offset:768
	global_load_b64 v[82:83], v[76:77], off offset:1280
	global_load_b64 v[76:77], v[76:77], off offset:1792
	v_add_nc_u32_e32 v72, 0x100, v86
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:8 offset1:9
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[76:77], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[76:77], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s31, v189
	v_add_nc_u32_e32 v72, s28, v192
	s_and_not1_b32 vcc_lo, exec_lo, s10
	s_movk_i32 s28, 0x1000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB7_9
; %bb.13:                               ; %.preheader501.i
                                        ;   in Loop: Header=BB7_10 Depth=2
	v_lshrrev_b32_e32 v195, v188, v194
	s_and_b32 s10, s30, exec_lo
	s_cselect_b32 s10, s17, 0x2400
	s_cselect_b32 s28, s19, 0x2c00
	v_cndmask_b32_e64 v196, 0, v193, s0
	v_cvt_f32_f16_e64 v195, v195.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v197, s10, v187
	v_add_nc_u32_e32 v198, s28, v187
	s_mov_b32 s28, 0
	v_cndmask_b32_e64 v195, 0, v195, s1
	ds_store_b64 v182, v[112:113]
	ds_store_b64 v183, v[114:115]
	ds_store_b32 v197, v196
	ds_store_b32 v198, v195
	s_branch .LBB7_9
.LBB7_14:                               ; %._crit_edge564.i
	v_mul_u32_u24_e32 v1, 0x500, v181
	v_lshlrev_b32_e32 v2, 2, v155
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v5, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v3, 0x50, v155
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x280, v180, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB7_105
; %bb.15:                               ; %.preheader497.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v155
	v_or_b32_e32 v8, s18, v5
	v_add3_u32 v6, 0, v3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB7_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	ds_load_b32 v12, v6
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off th:TH_STORE_NT
.LBB7_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:1280
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off th:TH_STORE_NT
.LBB7_19:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:2560
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128 th:TH_STORE_NT
.LBB7_21:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:3840
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128 th:TH_STORE_NT
.LBB7_23:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:5120
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256 th:TH_STORE_NT
.LBB7_25:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:6400
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256 th:TH_STORE_NT
.LBB7_27:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v13, v6 offset:7680
	global_load_b32 v12, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:384 th:TH_STORE_NT
.LBB7_29:
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v180
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB7_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v14, v6 offset:8960
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384 th:TH_STORE_NT
.LBB7_31:                               ; %.preheader495.1.i
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v170, v171 offset1:20
	ds_store_2addr_b32 v7, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB7_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s2, s20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v14, v6
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off th:TH_STORE_NT
.LBB7_33:
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_110
.LBB7_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_111
.LBB7_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_112
.LBB7_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_113
.LBB7_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_114
.LBB7_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_41
.LBB7_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:8960
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384 th:TH_STORE_NT
.LBB7_41:                               ; %.preheader495.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v162, v163 offset1:20
	ds_store_2addr_b32 v7, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB7_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	ds_load_b32 v16, v6
	global_load_b32 v15, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off th:TH_STORE_NT
.LBB7_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_116
.LBB7_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_117
.LBB7_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_118
.LBB7_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_119
.LBB7_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_120
.LBB7_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_51
.LBB7_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384 th:TH_STORE_NT
.LBB7_51:                               ; %.preheader495.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v153, v154 offset1:20
	ds_store_2addr_b32 v7, v151, v152 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v149, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v147, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB7_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	ds_load_b32 v18, v6
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v18, v17
	global_store_b32 v[15:16], v17, off th:TH_STORE_NT
.LBB7_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB7_121
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB7_122
.LBB7_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB7_123
.LBB7_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB7_124
.LBB7_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB7_125
.LBB7_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB7_126
.LBB7_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB7_61
.LBB7_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:8960
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384 th:TH_STORE_NT
.LBB7_61:                               ; %.preheader496.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v145, v146 offset1:20
	ds_store_2addr_b32 v7, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB7_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64 th:TH_STORE_NT
.LBB7_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB7_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:64 th:TH_STORE_NT
.LBB7_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB7_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192 th:TH_STORE_NT
.LBB7_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB7_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:192 th:TH_STORE_NT
.LBB7_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB7_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320 th:TH_STORE_NT
.LBB7_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB7_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:320 th:TH_STORE_NT
.LBB7_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB7_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v8, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	v_add_co_u32 v16, vcc_lo, v8, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	ds_load_b32 v18, v6 offset:7680
	global_load_b32 v8, v[16:17], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v18, v8
	global_store_b32 v[16:17], v8, off offset:448 th:TH_STORE_NT
.LBB7_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:8960
	global_load_b32 v16, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:448 th:TH_STORE_NT
.LBB7_77:                               ; %.preheader495.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v137, v138 offset1:20
	ds_store_2addr_b32 v7, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_128
.LBB7_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_129
.LBB7_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_130
.LBB7_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_131
.LBB7_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB7_132
.LBB7_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_133
.LBB7_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_86
.LBB7_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448 th:TH_STORE_NT
.LBB7_86:                               ; %.preheader495.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v129, v130 offset1:20
	ds_store_2addr_b32 v7, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_135
.LBB7_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_136
.LBB7_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_137
.LBB7_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_138
.LBB7_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_139
.LBB7_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_140
.LBB7_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_95
.LBB7_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:8960
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448 th:TH_STORE_NT
.LBB7_95:                               ; %.preheader495.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_142
.LBB7_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_143
.LBB7_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_144
.LBB7_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_145
.LBB7_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_146
.LBB7_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_147
.LBB7_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_104
.LBB7_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	ds_load_b32 v6, v6 offset:8960
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v7, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v8, v1, vcc_lo
	global_load_b32 v7, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	global_store_b32 v[0:1], v6, off offset:448 th:TH_STORE_NT
.LBB7_104:                              ; %.loopexit.loopexit597.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB7_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB7_107
; %bb.106:                              ; %.preheader494.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_lo_u32 v0, s12, v5
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[2:3], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_add3_u32 v3, 0, v3, v4
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v5, v0, v155, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, -1
	s_lshl_b32 s5, s12, 8
	s_movk_i32 s4, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v6, v5, s[0:3], null offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_or_b32 s6, s5, 0x80
	s_mul_i32 s7, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s8, s7, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v6
	buffer_store_b32 v0, v5, s[0:3], null offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s6, s5, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_movk_i32 s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s6, s5, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s6 offen
	s_lshl_b32 s4, s12, 6
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	s_add_co_i32 s6, s4, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v170, v171 offset1:20
	ds_store_2addr_b32 v2, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s8, s7, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_add_co_i32 s6, s4, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s8, s7, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_lshl_b32 s6, s12, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s6, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen th:TH_STORE_NT
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s10, s8, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v162, v163 offset1:20
	ds_store_2addr_b32 v2, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s10, s8, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_add_co_i32 s9, s6, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s10, s8, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_mul_i32 s9, s12, 0xc0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen th:TH_STORE_NT
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s12, s10, 0x80
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v153, v154 offset1:20
	ds_store_2addr_b32 v2, v151, v152 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v149, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v147, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s9 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s10, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_add_co_i32 s11, s9, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s12, s10, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_mov_b32 s11, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	s_or_b32 s12, s5, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v145, v146 offset1:20
	ds_store_2addr_b32 v2, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_or_b32 s12, s5, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s12, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s12 offen
	s_movk_i32 s11, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s12 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s11 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s11, s7, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	s_add_co_i32 s5, s4, 64
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v137, v138 offset1:20
	ds_store_2addr_b32 v2, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s11, s7, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_add_co_i32 s5, s4, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s5 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s11, s7, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s11 offen
	s_or_b32 s5, s8, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s11 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s7 offen
	s_or_b32 s4, s6, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s7 offen th:TH_STORE_NT
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v129, v130 offset1:20
	ds_store_2addr_b32 v2, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s8, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s6, 0x140
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s8, 0x140
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 64
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v4, v5, s[0:3], s6 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_add_co_i32 s5, s10, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v4
	buffer_store_b32 v0, v5, s[0:3], s6 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s8 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s8 offen th:TH_STORE_NT
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset1:5
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:10 offset1:15
	s_add_co_i32 s5, s10, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_add_co_i32 s4, s9, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v2, v5, s[0:3], s4 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:20 offset1:25
	s_add_co_i32 s5, s10, 0x140
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s4 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s5 offen th:TH_STORE_NT
	buffer_load_b32 v2, v5, s[0:3], s9 offen
	ds_load_2addr_stride64_b32 v[0:1], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v2
	buffer_store_b32 v0, v5, s[0:3], s9 offen th:TH_STORE_NT
	buffer_load_b32 v0, v5, s[0:3], s10 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v5, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB7_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB7_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1ELb1ELb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB7_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:1280
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_35
.LBB7_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:2560
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_36
.LBB7_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:3840
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_37
.LBB7_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:5120
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_38
.LBB7_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:6400
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB7_39
.LBB7_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	ds_load_b32 v15, v6 offset:7680
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB7_40
	s_branch .LBB7_41
.LBB7_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_45
.LBB7_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_46
.LBB7_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[14:15], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_47
.LBB7_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_48
.LBB7_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[14:15], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB7_49
.LBB7_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	ds_load_b32 v17, v6 offset:7680
	global_load_b32 v16, v[14:15], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[14:15], v16, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB7_50
	s_branch .LBB7_51
.LBB7_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:1280
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB7_55
.LBB7_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:2560
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB7_56
.LBB7_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:3840
	global_load_b32 v18, v[16:17], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB7_57
.LBB7_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:5120
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB7_58
.LBB7_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:6400
	global_load_b32 v18, v[16:17], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB7_59
.LBB7_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	ds_load_b32 v19, v6 offset:7680
	global_load_b32 v18, v[16:17], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v19, v18
	global_store_b32 v[16:17], v18, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB7_60
	s_branch .LBB7_61
.LBB7_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_79
.LBB7_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:1280
	global_load_b32 v16, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_80
.LBB7_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:2560
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_81
.LBB7_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:3840
	global_load_b32 v16, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_82
.LBB7_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:5120
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB7_83
.LBB7_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v17, v6 offset:6400
	global_load_b32 v16, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v17, v16
	global_store_b32 v[8:9], v16, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_84
.LBB7_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	ds_load_b32 v16, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v16, v10
	global_store_b32 v[8:9], v10, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_85
	s_branch .LBB7_86
.LBB7_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_88
.LBB7_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:1280
	global_load_b32 v10, v[8:9], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_89
.LBB7_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:2560
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_90
.LBB7_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:3840
	global_load_b32 v10, v[8:9], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_91
.LBB7_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:5120
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_92
.LBB7_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:6400
	global_load_b32 v10, v[8:9], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_93
.LBB7_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v11, v6 offset:7680
	global_load_b32 v10, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_94
	s_branch .LBB7_95
.LBB7_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_97
.LBB7_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:1280
	global_load_b32 v9, v[7:8], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_98
.LBB7_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:2560
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_99
.LBB7_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:3840
	global_load_b32 v9, v[7:8], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_100
.LBB7_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:5120
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_101
.LBB7_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:6400
	global_load_b32 v9, v[7:8], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB7_102
.LBB7_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v10, v6 offset:7680
	global_load_b32 v9, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB7_103
	s_branch .LBB7_104
.Lfunc_end7:
	.size	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add, .Lfunc_end7-gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 199
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end7-gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.num_vgpr, 199
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18572
; TotalNumSgprs: 40
; NumVgprs: 199
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 199
; Occupancy: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set ; -- Begin function gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
	.globl	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
	.p2align	8
	.type	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set,@function
gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set: ; @gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b96 s[12:14], s[0:1], 0x18
	s_lshl_b32 s16, ttmp9, 7
	s_lshl_b32 s18, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s16, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s18, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB8_108
; %bb.1:                                ; %.preheader501.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	s_add_co_i32 s2, s14, 0x7f
	s_ashr_i32 s1, s13, 31
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s10, s1, 24
	s_lshr_b32 s0, s3, 25
	s_add_co_i32 s10, s13, s10
	s_add_co_i32 s2, s2, s0
	s_lshr_b32 s0, s1, 26
	s_ashr_i32 s2, s2, 7
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s0, s0, 6
	s_lshl_b64 s[8:9], s[2:3], 12
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v3, 1, v0
	s_mul_u64 s[0:1], s[8:9], s[0:1]
	s_ashr_i32 s15, s10, 8
	s_add_co_i32 s17, s12, -1
	s_mul_i32 s10, s15, 0x88
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB8_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s18, v4
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v5, 2, v3
	s_mov_b32 s19, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v1
	s_cbranch_execz .LBB8_4
; %bb.3:
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 3, v[1:2]
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_b32 v2, v[1:2], off
.LBB8_4:                                ; %.preheader496.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s19
	v_or_b32_e32 v1, s16, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v6, s17, v1
	v_cmp_gt_i32_e32 vcc_lo, s12, v1
	v_mad_co_i64_i32 v[6:7], null, s10, v6, s[4:5]
	global_load_b32 v6, v[6:7], off
	v_lshlrev_b32_e32 v7, 4, v3
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v6, v7, v6
	v_lshlrev_b32_e32 v7, 3, v4
	v_cvt_f32_f16_e32 v6, v6.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v5, 0, v7, v5
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v6, vcc_lo
	ds_store_2addr_stride64_b32 v5, v2, v1 offset0:32 offset1:40
.LBB8_5:                                ; %Flow821
	s_or_b32 exec_lo, exec_lo, s11
	v_lshrrev_b32_e32 v7, 2, v0
	v_dual_mov_b32 v116, 0 :: v_dual_and_b32 v1, 3, v0
	v_dual_mov_b32 v148, 0 :: v_dual_lshlrev_b32 v9, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v2, s16, v7
	v_dual_mov_b32 v120, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v168, 0 :: v_dual_and_b32 v9, 16, v9
	v_dual_mov_b32 v118, 0 :: v_dual_add_nc_u32 v5, 64, v2
	v_min_i32_e32 v2, s17, v2
	v_lshrrev_b32_e32 v181, 5, v0
	v_bfe_u32 v8, v0, 1, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v5, s17, v5
	v_and_or_b32 v7, v7, 15, v9
	v_mad_co_u64_u32 v[68:69], null, s10, v2, v[1:2]
	v_or_b32_e32 v9, 8, v181
	v_and_or_b32 v10, v181, 6, v8
	v_mad_co_u64_u32 v[69:70], null, s10, v5, v[1:2]
	v_dual_mov_b32 v142, 0 :: v_dual_lshlrev_b32 v7, 3, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v8, v9, 14, v8
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v145, 15, v0
	v_mov_b32_e32 v122, 0
	s_clause 0x1
	global_load_b64 v[1:2], v68, s[4:5] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	v_lshl_or_b32 v9, v10, 8, v7
	v_lshl_or_b32 v7, v8, 8, v7
	v_mov_b32_e32 v137, 0
	v_bfe_u32 v180, v0, 4, 1
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_add_nc_u32_e32 v182, 0, v9
	v_add_nc_u32_e32 v183, 0, v7
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v178, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s11, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x1
	ds_store_b64 v182, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b64 v183, v[5:6]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB8_14
; %bb.6:                                ; %.preheader495.lr.ph.i
	v_dual_mov_b32 v172, 0 :: v_dual_and_b32 v1, 31, v0
	v_lshrrev_b32_e32 v2, 6, v0
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v7, s16, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v194, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v193, 0 :: v_dual_add_nc_u32 v6, s18, v4
	v_dual_mov_b32 v175, 0 :: v_dual_lshlrev_b32 v10, 2, v3
	v_lshl_or_b32 v9, v5, 11, v1
	v_lshl_or_b32 v184, v2, 10, v1
	v_min_i32_e32 v1, s17, v7
	v_dual_mov_b32 v173, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	s_add_co_i32 s13, s14, -1
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s10, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v70, s13, v6
	v_dual_mov_b32 v177, 0 :: v_dual_lshlrev_b32 v2, 6, v180
	v_add3_u32 v187, 0, v4, v10
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v4, 3, v145
	v_mad_co_u64_u32 v[66:67], null, s10, v1, v[66:67]
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v1, 9, v5
	v_add_co_u32 v185, s0, s0, v10
	v_add_co_u32 v190, s6, s6, v9
	v_ashrrev_i32_e32 v71, 31, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v186, null, s1, 0, s0
	v_cmp_gt_i32_e64 s0, s14, v6
	v_cmp_gt_i32_e64 s1, s12, v7
	v_dual_mov_b32 v143, 0 :: v_dual_lshlrev_b32 v188, 4, v3
	v_add3_u32 v189, 0, v8, v2
	v_add_co_ci_u32_e64 v191, null, s7, 0, s6
	v_add3_u32 v192, 0, v1, v4
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v116, 0
	s_mov_b32 s22, ttmp7
	s_mov_b32 s23, s11
	s_ashr_i32 s13, s14, 31
	s_movk_i32 s17, 0x2000
	s_movk_i32 s19, 0x2800
	s_mov_b32 s28, 0
	s_mov_b32 s6, s11
	s_branch .LBB8_8
.LBB8_7:                                ;   in Loop: Header=BB8_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_14
.LBB8_8:                                ; %.preheader495.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB8_10 Depth 2
	s_mov_b32 s7, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s26, s6, 1
	s_mul_u64 s[24:25], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s26, s15
	s_mov_b32 s31, 0
	s_cselect_b32 s27, -1, 0
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_mov_b32 s30, 0
	s_mov_b32 s29, -1
	s_branch .LBB8_10
.LBB8_9:                                ;   in Loop: Header=BB8_10 Depth=2
	s_wait_dscnt 0xb
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v59, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v114, v109, v94 :: v_dual_mul_f32 v57, v112, v57
	v_mul_f32_e32 v112, v111, v94
	v_mul_f32_e32 v58, v113, v58
	v_mul_f32_e32 v113, v106, v94
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_fmac_f32_e32 v57, v112, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v112, v104, v94 :: v_dual_mul_f32 v59, v113, v59
	v_dual_mul_f32 v113, v107, v94 :: v_dual_fmac_f32 v58, v114, v95
	v_add_f32_e32 v172, v172, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v57, v112, v60 :: v_dual_mul_f32 v60, v105, v94
	v_dual_fmac_f32 v59, v113, v95 :: v_dual_mul_f32 v112, v100, v94
	v_mul_f32_e32 v113, v102, v94
	v_cvt_f32_i32_e32 v62, v62
	v_add_f32_e32 v179, v179, v58
	v_fmac_f32_e32 v57, v60, v95
	v_dual_add_f32 v177, v177, v59 :: v_dual_mul_f32 v60, v96, v94
	v_mul_f32_e32 v58, v112, v61
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v59, v113, v62
	v_dual_mul_f32 v62, v101, v94 :: v_dual_mul_f32 v63, v103, v94
	v_mul_f32_e32 v112, v98, v94
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v60, v60, v61 :: v_dual_mul_f32 v61, v97, v94
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v58, v62, v95 :: v_dual_fmac_f32 v59, v63, v95
	v_dual_mul_f32 v62, v112, v64 :: v_dual_mul_f32 v63, v99, v94
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v60, v61, v95
	v_dual_add_f32 v178, v178, v57 :: v_dual_add_f32 v175, v175, v58
	s_wait_dscnt 0xa
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_dual_add_f32 v176, v176, v59 :: v_dual_add_f32 v173, v173, v60
	v_add_f32_e32 v174, v174, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
	v_dual_add_f32 v170, v170, v49 :: v_dual_mul_f32 v49, v57, v52
	v_dual_mul_f32 v57, v100, v92 :: v_dual_mul_f32 v52, v105, v92
	v_fmac_f32_e32 v51, v58, v59
	v_dual_mul_f32 v58, v102, v92 :: v_dual_add_f32 v171, v171, v50
	v_cvt_f32_i32_e32 v56, v56
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v49, v52, v59
	v_dual_add_f32 v169, v169, v51 :: v_dual_mul_f32 v52, v96, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v50, v57, v53 :: v_dual_mul_f32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v54, v101, v92 :: v_dual_mul_f32 v55, v103, v92
	v_mul_f32_e32 v57, v98, v92
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v50, v54, v59 :: v_dual_fmac_f32 v51, v55, v59
	v_mul_f32_e32 v54, v57, v56
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v55, v99, v92 :: v_dual_fmac_f32 v52, v53, v59
	v_dual_add_f32 v168, v168, v49 :: v_dual_add_f32 v167, v167, v51
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_add_f32_e32 v166, v166, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_dual_mul_f32 v49, v111, v90 :: v_dual_add_f32 v164, v164, v52
	v_add_f32_e32 v165, v165, v54
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_add_f32_e32 v162, v162, v41
	v_cvt_f32_i32_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v163, v163, v42 :: v_dual_add_f32 v160, v160, v43
	v_mul_f32_e32 v42, v49, v45
	v_cvt_f32_i32_e32 v45, v47
	v_dual_mul_f32 v47, v103, v90 :: v_dual_mul_f32 v50, v102, v90
	v_fmac_f32_e32 v41, v44, v51
	v_dual_mul_f32 v44, v96, v90 :: v_dual_mul_f32 v49, v98, v90
	v_cvt_f32_i32_e32 v48, v48
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v46
	v_mul_f32_e32 v46, v101, v90
	v_dual_mul_f32 v44, v44, v45 :: v_dual_mul_f32 v45, v97, v90
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v43, v47, v51 :: v_dual_fmac_f32 v42, v46, v51
	v_dual_mul_f32 v47, v99, v90 :: v_dual_mul_f32 v46, v49, v48
	v_fmac_f32_e32 v44, v45, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_add_f32 v161, v161, v41 :: v_dual_add_f32 v158, v158, v42
	s_wait_dscnt 0x8
	v_dual_mul_f32 v41, v110, v72 :: v_dual_mul_f32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_add_f32_e32 v159, v159, v43
	v_dual_fmac_f32 v46, v47, v51 :: v_dual_mul_f32 v33, v41, v33
	v_cvt_f32_i32_e32 v43, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v41, v111, v72 :: v_dual_mul_f32 v34, v42, v34
	v_mul_f32_e32 v42, v106, v72
	v_cvt_f32_i32_e32 v35, v35
	v_dual_add_f32 v156, v156, v44 :: v_dual_add_f32 v157, v157, v46
	v_fmac_f32_e32 v33, v41, v43
	v_dual_mul_f32 v41, v104, v72 :: v_dual_mul_f32 v44, v109, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v35, v42, v35
	v_cvt_f32_i32_e32 v36, v36
	v_mul_f32_e32 v42, v107, v72
	v_cvt_f32_i32_e32 v37, v37
	v_fmac_f32_e32 v34, v44, v43
	v_dual_add_f32 v154, v154, v33 :: v_dual_mul_f32 v33, v41, v36
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v36, v105, v72
	v_fmac_f32_e32 v35, v42, v43
	v_mul_f32_e32 v42, v102, v72
	v_cvt_f32_i32_e32 v38, v38
	v_add_f32_e32 v155, v155, v34
	v_fmac_f32_e32 v33, v36, v43
	v_add_f32_e32 v152, v152, v35
	v_dual_mul_f32 v34, v41, v37 :: v_dual_mul_f32 v37, v103, v72
	v_dual_mul_f32 v35, v42, v38 :: v_dual_mul_f32 v36, v101, v72
	v_dual_mul_f32 v41, v98, v72 :: v_dual_mul_f32 v38, v96, v72
	v_add_f32_e32 v153, v153, v33
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_dual_fmac_f32 v34, v36, v43 :: v_dual_fmac_f32 v35, v37, v43
	v_mul_f32_e32 v37, v97, v72
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v33, v38, v39 :: v_dual_mul_f32 v36, v41, v40
	v_mul_f32_e32 v38, v99, v72
	s_wait_dscnt 0x6
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_add_f32 v150, v150, v34 :: v_dual_fmac_f32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_mul_f32 v25, v39, v25
	v_dual_mul_f32 v26, v40, v26 :: v_dual_mul_f32 v37, v94, v87
	v_dual_mul_f32 v34, v94, v89 :: v_dual_add_f32 v151, v151, v35
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v148, v148, v33 :: v_dual_add_f32 v149, v149, v36
	v_dual_fmac_f32 v26, v37, v95 :: v_dual_fmac_f32 v25, v34, v95
	s_wait_dscnt 0x4
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_add_f32 v146, v146, v25 :: v_dual_add_f32 v147, v147, v26
	v_mul_f32_e32 v26, v94, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	s_wait_dscnt 0x3
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	s_wait_dscnt 0x2
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v143, v143, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v144, v144, v27
	s_wait_dscnt 0x1
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	s_wait_dscnt 0x0
	v_mul_f32_e32 v30, v94, v76
	v_add_f32_e32 v142, v142, v28
	v_mul_f32_e32 v28, v94, v75
	v_fmac_f32_e32 v25, v26, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v26, v29, v31 :: v_dual_mul_f32 v27, v30, v32
	v_dual_mul_f32 v29, v94, v77 :: v_dual_mul_f32 v30, v92, v88
	v_mul_f32_e32 v31, v92, v86
	v_cvt_f32_i32_e32 v18, v18
	v_dual_add_f32 v141, v141, v25 :: v_dual_fmac_f32 v26, v28, v95
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v27, v29, v95
	v_dual_mul_f32 v17, v30, v17 :: v_dual_mul_f32 v28, v92, v87
	v_mul_f32_e32 v18, v31, v18
	v_mul_f32_e32 v30, v92, v84
	v_cvt_f32_i32_e32 v20, v20
	v_mul_f32_e32 v25, v92, v89
	v_mul_f32_e32 v29, v92, v82
	v_cvt_f32_i32_e32 v19, v19
	v_dual_add_f32 v139, v139, v26 :: v_dual_fmac_f32 v18, v28, v59
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v20, v30, v20 :: v_dual_fmac_f32 v17, v25, v59
	v_dual_mul_f32 v26, v92, v85 :: v_dual_mul_f32 v25, v92, v83
	v_add_f32_e32 v138, v138, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v140, v140, v27 :: v_dual_add_f32 v137, v137, v17
	v_dual_fmac_f32 v20, v26, v59 :: v_dual_mul_f32 v17, v92, v80
	v_mul_f32_e32 v19, v29, v19
	v_cvt_f32_i32_e32 v18, v21
	v_mul_f32_e32 v21, v92, v78
	v_cvt_f32_i32_e32 v22, v22
	v_add_f32_e32 v136, v136, v20
	v_mul_f32_e32 v20, v92, v74
	v_dual_mul_f32 v17, v17, v18 :: v_dual_mul_f32 v18, v92, v81
	v_fmac_f32_e32 v19, v25, v59
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v11, v11
	v_add_f32_e32 v135, v135, v19
	v_mul_f32_e32 v19, v21, v22
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v22, v92, v79
	v_cvt_f32_i32_e32 v23, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_mul_f32_e32 v20, v20, v21
	v_mul_f32_e32 v21, v92, v75
	v_dual_fmac_f32 v17, v18, v59 :: v_dual_mul_f32 v18, v92, v76
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v20, v21, v59 :: v_dual_mul_f32 v21, v90, v88
	v_add_f32_e32 v133, v133, v17
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v131, v131, v20 :: v_dual_mul_f32 v20, v90, v82
	v_mul_f32_e32 v9, v21, v9
	v_mul_f32_e32 v21, v90, v84
	v_dual_mul_f32 v17, v18, v23 :: v_dual_mul_f32 v18, v92, v77
	v_fmac_f32_e32 v19, v22, v59
	v_mul_f32_e32 v22, v90, v86
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v12, v21, v12 :: v_dual_mul_f32 v11, v20, v11
	v_mul_f32_e32 v20, v90, v78
	v_fmac_f32_e32 v17, v18, v59
	v_mul_f32_e32 v10, v22, v10
	v_add_f32_e32 v134, v134, v19
	v_dual_mul_f32 v18, v90, v89 :: v_dual_mul_f32 v19, v90, v87
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v10, v19, v51 :: v_dual_fmac_f32 v9, v18, v51
	v_dual_mul_f32 v18, v90, v85 :: v_dual_mul_f32 v19, v90, v80
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v130, v130, v10
	v_mul_f32_e32 v10, v90, v74
	v_dual_fmac_f32 v12, v18, v51 :: v_dual_add_f32 v129, v129, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v9, v19, v13
	v_dual_mul_f32 v13, v20, v14 :: v_dual_mul_f32 v14, v90, v81
	v_add_f32_e32 v128, v128, v12
	v_dual_add_f32 v132, v132, v17 :: v_dual_mul_f32 v17, v90, v83
	v_mul_f32_e32 v12, v90, v76
	s_barrier_signal -1
	s_xor_b32 s10, s29, -1
	s_mov_b32 s31, 1
	v_fmac_f32_e32 v11, v17, v51
	v_mul_f32_e32 v17, v90, v79
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s10
	s_mov_b32 s30, -1
	v_add_f32_e32 v127, v127, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v126, v126, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_add_f32_e32 v125, v125, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v1, v12, v1
	v_mul_f32_e32 v12, v72, v89
	v_fmac_f32_e32 v1, v12, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v12, v72, v84 :: v_dual_fmac_f32 v9, v10, v51
	v_dual_mul_f32 v10, v72, v86 :: v_dual_add_f32 v121, v121, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v2, v10, v2
	v_mul_f32_e32 v10, v72, v82
	v_dual_mul_f32 v1, v10, v3 :: v_dual_mul_f32 v10, v72, v80
	v_add_f32_e32 v123, v123, v9
	v_mul_f32_e32 v9, v72, v87
	v_mul_f32_e32 v3, v12, v4
	v_mul_f32_e32 v4, v72, v83
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v9, v43
	v_fmac_f32_e32 v11, v13, v51
	v_dual_mul_f32 v9, v72, v85 :: v_dual_add_f32 v122, v122, v2
	v_mul_f32_e32 v2, v10, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v124, v124, v11
	v_dual_mul_f32 v11, v72, v78 :: v_dual_mul_f32 v10, v72, v79
	v_fmac_f32_e32 v3, v9, v43
	v_mul_f32_e32 v9, v72, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v120, v120, v3
	v_dual_fmac_f32 v1, v4, v43 :: v_dual_mul_f32 v4, v11, v6
	v_mul_f32_e32 v6, v72, v74
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v4, v10, v43
	v_dual_mul_f32 v6, v6, v7 :: v_dual_mul_f32 v7, v9, v8
	v_dual_mul_f32 v8, v72, v75 :: v_dual_mul_f32 v5, v72, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_add_f32 v118, v118, v4 :: v_dual_mul_f32 v9, v72, v77
	v_dual_add_f32 v119, v119, v1 :: v_dual_fmac_f32 v6, v8, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v2, v5, v43
	v_dual_add_f32 v116, v116, v6 :: v_dual_fmac_f32 v7, v9, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v117, v117, v2
	v_add_f32_e32 v67, v67, v7
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_7
.LBB8_10:                               ;   Parent Loop BB8_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s33, s31, s7
	v_add_nc_u32_e32 v86, s28, v184
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s10, s33, 1
	s_and_b32 s28, s30, s27
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[10:11], s[2:3]
	s_lshl_b32 s10, s31, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[22:23]
	ds_load_2addr_stride64_b64 v[74:77], v86 offset1:1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[34:35], s[34:35], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, vcc_lo, v190, s34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, s35, v191, vcc_lo
	s_add_nc_u64 s[34:35], s[24:25], s[10:11]
	s_clause 0x1
	global_load_b64 v[1:2], v[72:73], off
	global_load_b64 v[3:4], v[72:73], off offset:512
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v7, s10, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s10
	v_add_co_u32 v9, s10, s34, v69
	s_clause 0x1
	global_load_b64 v[5:6], v[72:73], off offset:1024
	global_load_b64 v[78:79], v[72:73], off offset:1536
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s35, 0, s10
	s_clause 0x1
	global_load_b64 v[112:113], v[7:8], off offset:40
	global_load_b64 v[114:115], v[9:10], off offset:40
	s_wait_loadcnt_dscnt 0x500
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[1:2], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[3:4], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[1:2], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[3:4], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[5:6], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[78:79], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[5:6], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[78:79], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[72:73], off offset:256
	global_load_b64 v[80:81], v[72:73], off offset:768
	global_load_b64 v[82:83], v[72:73], off offset:1280
	global_load_b64 v[84:85], v[72:73], off offset:1792
	ds_load_2addr_b64 v[74:77], v86 offset0:32 offset1:96
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[74:75], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[74:75], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[74:75], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[74:75], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[76:77], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[76:77], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[76:77], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[76:77], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_store_b64 v182, v[112:113] offset:4096
	ds_store_b64 v183, v[114:115] offset:4096
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_12
; %bb.11:                               ;   in Loop: Header=BB8_10 Depth=2
	s_add_co_i32 s33, s33, 1
	s_add_co_i32 s10, s31, s6
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[74:75], null, s33, s14, v[70:71]
	s_mul_u64 s[34:35], s[10:11], 0x88
	s_xor_b32 s31, s31, 1
	s_mov_b32 s37, s11
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s31, 6
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_mulk_i32 s10, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_mad_co_u64_u32 v[75:76], null, s33, s13, v[75:76]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v76, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v77, null, s35, 0, s33
	v_add_co_u32 v78, s33, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s33
	v_lshlrev_b64_e32 v[74:75], 3, v[74:75]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v74, vcc_lo, v185, v74
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v75, null, v186, v75, vcc_lo
	v_add_co_u32 v80, vcc_lo, v65, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v66, vcc_lo
	s_lshl_b32 s10, s31, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, vcc_lo, v80, s10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[112:113], v[76:77], off offset:8
	global_load_b64 v[114:115], v[78:79], off offset:8
	global_load_b32 v193, v[74:75], off
	global_load_b32 v194, v[80:81], off
.LBB8_12:                               ; %.preheader493.i
                                        ;   in Loop: Header=BB8_10 Depth=2
	v_add_co_u32 v76, vcc_lo, v72, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, s9, v73, vcc_lo
	v_add_nc_u32_e32 v86, 0, v184
	s_xor_b32 s10, s28, -1
	s_and_b32 s28, s29, exec_lo
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off
	global_load_b64 v[80:81], v[76:77], off offset:512
	global_load_b64 v[82:83], v[76:77], off offset:1024
	global_load_b64 v[84:85], v[76:77], off offset:1536
	s_cselect_b32 s28, s17, 0x2400
	ds_load_2addr_stride64_b64 v[72:75], v86 offset0:8 offset1:9
	s_cselect_b32 s31, s19, 0x2c00
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[84:85], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[84:85], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_clause 0x3
	global_load_b64 v[78:79], v[76:77], off offset:256
	global_load_b64 v[80:81], v[76:77], off offset:768
	global_load_b64 v[82:83], v[76:77], off offset:1280
	global_load_b64 v[76:77], v[76:77], off offset:1792
	v_add_nc_u32_e32 v72, 0x100, v86
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:8 offset1:9
	s_wait_loadcnt_dscnt 0x300
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[78:79], v[57:64] neg_lo:[0,1,0]
	s_wait_loadcnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[80:81], v[49:56] neg_lo:[0,1,0]
	s_wait_loadcnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[82:83], v[41:48] neg_lo:[0,1,0]
	s_wait_loadcnt 0x0
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[76:77], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[78:79], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[80:81], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[82:83], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[76:77], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s31, v189
	v_add_nc_u32_e32 v72, s28, v192
	s_and_not1_b32 vcc_lo, exec_lo, s10
	s_movk_i32 s28, 0x1000
	ds_load_2addr_b32 v[110:111], v76 offset1:1
	ds_load_2addr_b32 v[108:109], v76 offset0:2 offset1:3
	ds_load_2addr_b32 v[106:107], v76 offset0:4 offset1:5
	ds_load_2addr_b32 v[104:105], v76 offset0:6 offset1:7
	ds_load_2addr_b32 v[100:101], v76 offset0:8 offset1:9
	ds_load_2addr_b32 v[102:103], v76 offset0:10 offset1:11
	ds_load_2addr_b32 v[96:97], v76 offset0:12 offset1:13
	ds_load_2addr_b32 v[98:99], v76 offset0:14 offset1:15
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[90:91], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	ds_load_2addr_b32 v[88:89], v76 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v76 offset0:34 offset1:35
	ds_load_2addr_b32 v[82:83], v76 offset0:36 offset1:37
	ds_load_2addr_b32 v[84:85], v76 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v76 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v76 offset0:42 offset1:43
	ds_load_2addr_b32 v[74:75], v76 offset0:44 offset1:45
	ds_load_2addr_b32 v[76:77], v76 offset0:46 offset1:47
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_9
; %bb.13:                               ; %.preheader494.i
                                        ;   in Loop: Header=BB8_10 Depth=2
	v_lshrrev_b32_e32 v195, v188, v194
	s_and_b32 s10, s30, exec_lo
	s_cselect_b32 s10, s17, 0x2400
	s_cselect_b32 s28, s19, 0x2c00
	v_cndmask_b32_e64 v196, 0, v193, s0
	v_cvt_f32_f16_e64 v195, v195.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v197, s10, v187
	v_add_nc_u32_e32 v198, s28, v187
	s_mov_b32 s28, 0
	v_cndmask_b32_e64 v195, 0, v195, s1
	ds_store_b64 v182, v[112:113]
	ds_store_b64 v183, v[114:115]
	ds_store_b32 v197, v196
	ds_store_b32 v198, v195
	s_branch .LBB8_9
.LBB8_14:                               ; %._crit_edge557.i
	v_mul_u32_u24_e32 v1, 0x500, v181
	v_lshlrev_b32_e32 v2, 2, v145
	s_add_co_i32 s0, s16, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_lshrrev_b32_e32 v3, 4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s18, 0x80
	v_add3_u32 v7, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s1, s14
	v_mul_u32_u24_e32 v4, 0x50, v145
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v3
	v_mad_u32_u24 v2, 0x280, v180, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB8_105
; %bb.15:                               ; %.preheader490.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v145
	v_or_b32_e32 v8, s18, v3
	v_add3_u32 v6, 0, v4, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e64 s7, s12, v0
	v_cmp_gt_i32_e32 vcc_lo, s14, v8
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB8_17
; %bb.16:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v6
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, s20, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, v9, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off th:TH_STORE_NT
.LBB8_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_19
; %bb.18:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off th:TH_STORE_NT
.LBB8_19:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_21
; %bb.20:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128 th:TH_STORE_NT
.LBB8_21:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_23
; %bb.22:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128 th:TH_STORE_NT
.LBB8_23:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_25
; %bb.24:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256 th:TH_STORE_NT
.LBB8_25:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_27
; %bb.26:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v6 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256 th:TH_STORE_NT
.LBB8_27:
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_29
; %bb.28:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v6 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, s20, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384 th:TH_STORE_NT
.LBB8_29:
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v180
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB8_31
; %bb.30:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	ds_load_b32 v15, v6 offset:8960
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, s20, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:384 th:TH_STORE_NT
.LBB8_31:                               ; %.preheader488.1.i
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v7, v7, v10
	v_or_b32_e32 v10, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v10
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v170, v171 offset1:20
	ds_store_2addr_b32 v7, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB8_33
; %bb.32:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v6
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v11, s2, s20, v11
	v_add_co_ci_u32_e64 v12, null, s21, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off th:TH_STORE_NT
.LBB8_33:
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_110
.LBB8_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_111
.LBB8_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_112
.LBB8_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_113
.LBB8_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_114
.LBB8_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_41
.LBB8_40:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384 th:TH_STORE_NT
.LBB8_41:                               ; %.preheader488.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 32, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v12
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v162, v163 offset1:20
	ds_store_2addr_b32 v7, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB8_43
; %bb.42:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	ds_load_b32 v17, v6
	v_lshlrev_b64_e32 v[15:16], 2, v[0:1]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, s20, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s4, v13, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s4
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v17, off th:TH_STORE_NT
.LBB8_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_116
.LBB8_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_117
.LBB8_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_118
.LBB8_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_119
.LBB8_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_120
.LBB8_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_51
.LBB8_50:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384 th:TH_STORE_NT
.LBB8_51:                               ; %.preheader488.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v14, 48, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v14
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v154, v155 offset1:20
	ds_store_2addr_b32 v7, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v150, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v148, v149 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB8_53
; %bb.52:
	v_mad_co_i64_i32 v[15:16], null, s12, v14, 0
	ds_load_b32 v19, v6
	v_lshlrev_b64_e32 v[17:18], 2, v[0:1]
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, s20, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, s21, v16, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v15, s6, v15, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, v16, v18, s6
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v19, off th:TH_STORE_NT
.LBB8_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB8_121
; %bb.54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB8_122
.LBB8_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB8_123
.LBB8_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB8_124
.LBB8_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB8_125
.LBB8_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB8_126
.LBB8_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB8_61
.LBB8_60:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:8960
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384 th:TH_STORE_NT
.LBB8_61:                               ; %.preheader489.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v16, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v16
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v146, v147 offset1:20
	ds_store_2addr_b32 v7, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB8_63
; %bb.62:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64 th:TH_STORE_NT
.LBB8_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB8_65
; %bb.64:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s8, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s8
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:64 th:TH_STORE_NT
.LBB8_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB8_67
; %bb.66:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192 th:TH_STORE_NT
.LBB8_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB8_69
; %bb.68:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s9, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s9
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:192 th:TH_STORE_NT
.LBB8_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB8_71
; %bb.70:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320 th:TH_STORE_NT
.LBB8_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB8_73
; %bb.72:
	v_mad_co_i64_i32 v[16:17], null, s12, v9, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s10, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s10
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:320 th:TH_STORE_NT
.LBB8_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB8_75
; %bb.74:
	v_mad_co_i64_i32 v[16:17], null, s12, v8, 0
	ds_load_b32 v8, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, s20, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, vcc_lo, v16, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v8, off offset:448 th:TH_STORE_NT
.LBB8_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_77
; %bb.76:
	v_mad_co_i64_i32 v[8:9], null, s12, v9, 0
	ds_load_b32 v18, v6 offset:8960
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:448 th:TH_STORE_NT
.LBB8_77:                               ; %.preheader488.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v137, v138 offset1:20
	ds_store_2addr_b32 v7, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_128
.LBB8_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_129
.LBB8_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_130
.LBB8_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_131
.LBB8_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB8_132
.LBB8_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_133
.LBB8_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_86
.LBB8_85:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:448 th:TH_STORE_NT
.LBB8_86:                               ; %.preheader488.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v129, v130 offset1:20
	ds_store_2addr_b32 v7, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_135
.LBB8_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_136
.LBB8_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_137
.LBB8_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_138
.LBB8_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_139
.LBB8_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_140
.LBB8_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_95
.LBB8_94:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v12, v6 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448 th:TH_STORE_NT
.LBB8_95:                               ; %.preheader488.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_142
.LBB8_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_143
.LBB8_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_144
.LBB8_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_145
.LBB8_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_146
.LBB8_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_147
.LBB8_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_104
.LBB8_103:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v9, v6 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s20, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s21, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v6, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v7, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v9, off offset:448 th:TH_STORE_NT
.LBB8_104:                              ; %.loopexit.loopexit590.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB8_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB8_107
; %bb.106:                              ; %.preheader487.i
	ds_store_2addr_b32 v2, v172, v179 offset1:20
	ds_store_2addr_b32 v2, v177, v178 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v175, v176 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v173, v174 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add3_u32 v10, 0, v4, v5
	v_mul_lo_u32 v3, s12, v3
	s_ashr_i32 s19, s18, 31
	s_ashr_i32 s17, s16, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[18:19]
	s_lshl_b64 s[4:5], s[16:17], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_mov_b32 s3, 0x31004000
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_add_lshl_u32 v11, v3, v145, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_mov_b32 s2, -1
	s_lshl_b32 s6, s12, 8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s1, s1, 0xffff
	s_movk_i32 s7, 0x80
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[8:9], v10 offset0:30 offset1:35
	s_movk_i32 s8, 0x100
	s_movk_i32 s9, 0x180
	s_or_b32 s10, s6, 0x80
	s_add_co_i32 s4, s6, 0x100
	s_add_co_i32 s5, s6, 0x180
	s_add_co_i32 s17, s6, 0x140
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], null offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s6 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s7 offen th:TH_STORE_NT
	buffer_store_b32 v5, v11, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s8 offen th:TH_STORE_NT
	buffer_store_b32 v7, v11, s[0:3], s4 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v8, v11, s[0:3], s9 offen th:TH_STORE_NT
	buffer_store_b32 v9, v11, s[0:3], s5 offen th:TH_STORE_NT
	s_lshl_b32 s4, s12, 6
	s_mul_i32 s5, s12, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s4, 0x80
	s_add_co_i32 s8, s5, 0x80
	s_add_co_i32 s9, s4, 0x100
	s_add_co_i32 s10, s5, 0x100
	s_add_co_i32 s11, s4, 0x180
	s_add_co_i32 s13, s5, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v170, v171 offset1:20
	ds_store_2addr_b32 v2, v169, v168 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v166, v167 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v164, v165 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s5 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s7 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s8 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s9 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s11 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s13 offen th:TH_STORE_NT
	s_lshl_b32 s7, s12, 7
	s_mul_i32 s8, s12, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s7, 0x80
	s_add_co_i32 s10, s8, 0x80
	s_add_co_i32 s11, s7, 0x100
	s_add_co_i32 s13, s8, 0x100
	s_add_co_i32 s14, s7, 0x180
	s_add_co_i32 s15, s8, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v162, v163 offset1:20
	ds_store_2addr_b32 v2, v160, v161 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v158, v159 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v156, v157 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s7 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s8 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s9 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s13 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s14 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s15 offen th:TH_STORE_NT
	s_mul_i32 s9, s12, 0xc0
	s_mul_i32 s10, s12, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s11, s9, 0x80
	s_add_co_i32 s12, s10, 0x80
	s_add_co_i32 s13, s9, 0x100
	s_add_co_i32 s14, s10, 0x100
	s_add_co_i32 s15, s9, 0x180
	s_add_co_i32 s16, s10, 0x180
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v154, v155 offset1:20
	ds_store_2addr_b32 v2, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v150, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v148, v149 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s9 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s11 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s12 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s13 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s14 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s15 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s16 offen th:TH_STORE_NT
	s_mov_b32 s14, 64
	s_or_b32 s15, s6, 64
	s_movk_i32 s11, 0x140
	s_movk_i32 s12, 0xc0
	s_movk_i32 s13, 0x1c0
	s_or_b32 s16, s6, 0xc0
	s_addk_co_i32 s6, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v146, v147 offset1:20
	ds_store_2addr_b32 v2, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s14 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s15 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s16 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s11 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s17 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s13 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s6 offen th:TH_STORE_NT
	s_add_co_i32 s6, s4, 64
	s_add_co_i32 s11, s5, 64
	s_add_co_i32 s12, s4, 0xc0
	s_add_co_i32 s13, s5, 0xc0
	s_add_co_i32 s14, s4, 0x140
	s_add_co_i32 s15, s5, 0x140
	s_addk_co_i32 s4, 0x1c0
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v137, v138 offset1:20
	ds_store_2addr_b32 v2, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s6 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s11 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s12 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s13 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s14 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s15 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s4 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s5 offen th:TH_STORE_NT
	s_or_b32 s4, s7, 64
	s_or_b32 s5, s8, 64
	s_add_co_i32 s6, s7, 0xc0
	s_add_co_i32 s11, s8, 0xc0
	s_add_co_i32 s12, s7, 0x140
	s_add_co_i32 s13, s8, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_addk_co_i32 s8, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v129, v130 offset1:20
	ds_store_2addr_b32 v2, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v125, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[3:4], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[5:6], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[7:8], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s5 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v3, v11, s[0:3], s6 offen th:TH_STORE_NT
	buffer_store_b32 v4, v11, s[0:3], s11 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v5, v11, s[0:3], s12 offen th:TH_STORE_NT
	buffer_store_b32 v6, v11, s[0:3], s13 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v7, v11, s[0:3], s7 offen th:TH_STORE_NT
	buffer_store_b32 v8, v11, s[0:3], s8 offen th:TH_STORE_NT
	s_add_co_i32 s4, s9, 64
	s_add_co_i32 s5, s10, 64
	s_add_co_i32 s6, s9, 0xc0
	s_add_co_i32 s7, s10, 0xc0
	s_add_co_i32 s8, s9, 0x140
	s_add_co_i32 s11, s10, 0x140
	s_addk_co_i32 s9, 0x1c0
	s_addk_co_i32 s10, 0x1c0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v119, v120 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v116, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_2addr_stride64_b32 v[0:1], v10 offset1:5
	ds_load_2addr_stride64_b32 v[2:3], v10 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[4:5], v10 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[6:7], v10 offset0:30 offset1:35
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v11, s[0:3], s4 offen th:TH_STORE_NT
	buffer_store_b32 v1, v11, s[0:3], s5 offen th:TH_STORE_NT
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v2, v11, s[0:3], s6 offen th:TH_STORE_NT
	buffer_store_b32 v3, v11, s[0:3], s7 offen th:TH_STORE_NT
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v4, v11, s[0:3], s8 offen th:TH_STORE_NT
	buffer_store_b32 v5, v11, s[0:3], s11 offen th:TH_STORE_NT
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v6, v11, s[0:3], s9 offen th:TH_STORE_NT
	buffer_store_b32 v7, v11, s[0:3], s10 offen th:TH_STORE_NT
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB8_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB8_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0ELb1ELb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB8_109:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_35
.LBB8_110:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_36
.LBB8_111:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_37
.LBB8_112:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_38
.LBB8_113:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB8_39
.LBB8_114:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v6 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, s20, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s3, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s3
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB8_40
	s_branch .LBB8_41
.LBB8_115:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_45
.LBB8_116:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_46
.LBB8_117:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_47
.LBB8_118:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_48
.LBB8_119:
	v_mad_co_i64_i32 v[14:15], null, s12, v13, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB8_49
.LBB8_120:
	v_mad_co_i64_i32 v[14:15], null, s12, v12, 0
	ds_load_b32 v18, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, s20, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, s21, v15, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v14, s5, v14, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, v15, v17, s5
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v18, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB8_50
	s_branch .LBB8_51
.LBB8_121:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:1280
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB8_55
.LBB8_122:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:2560
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB8_56
.LBB8_123:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:3840
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:128 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB8_57
.LBB8_124:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:5120
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB8_58
.LBB8_125:
	v_mad_co_i64_i32 v[16:17], null, s12, v15, 0
	ds_load_b32 v20, v6 offset:6400
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:256 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB8_59
.LBB8_126:
	v_mad_co_i64_i32 v[16:17], null, s12, v14, 0
	ds_load_b32 v20, v6 offset:7680
	v_lshlrev_b64_e32 v[18:19], 2, v[0:1]
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, s20, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, s21, v17, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v16, s7, v16, v18
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, v17, v19, s7
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v20, off offset:384 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB8_60
	s_branch .LBB8_61
.LBB8_127:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_79
.LBB8_128:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:1280
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_80
.LBB8_129:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:2560
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_81
.LBB8_130:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:3840
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_82
.LBB8_131:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v18, v6 offset:5120
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB8_83
.LBB8_132:
	v_mad_co_i64_i32 v[8:9], null, s12, v11, 0
	ds_load_b32 v18, v6 offset:6400
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v18, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_84
.LBB8_133:
	v_mad_co_i64_i32 v[8:9], null, s12, v10, 0
	ds_load_b32 v10, v6 offset:7680
	v_lshlrev_b64_e32 v[16:17], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v10, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_85
	s_branch .LBB8_86
.LBB8_134:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_88
.LBB8_135:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:1280
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_89
.LBB8_136:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:2560
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_90
.LBB8_137:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:3840
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_91
.LBB8_138:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v16, v6 offset:5120
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_92
.LBB8_139:
	v_mad_co_i64_i32 v[8:9], null, s12, v13, 0
	ds_load_b32 v16, v6 offset:6400
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v16, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_93
.LBB8_140:
	v_mad_co_i64_i32 v[8:9], null, s12, v12, 0
	ds_load_b32 v12, v6 offset:7680
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, s20, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s21, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_94
	s_branch .LBB8_95
.LBB8_141:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_97
.LBB8_142:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:1280
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_98
.LBB8_143:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:2560
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_99
.LBB8_144:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:3840
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:192 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_100
.LBB8_145:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:5120
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_101
.LBB8_146:
	v_mad_co_i64_i32 v[7:8], null, s12, v15, 0
	ds_load_b32 v11, v6 offset:6400
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:320 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB8_102
.LBB8_147:
	v_mad_co_i64_i32 v[7:8], null, s12, v14, 0
	ds_load_b32 v11, v6 offset:7680
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, s20, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s21, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448 th:TH_STORE_NT
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB8_103
	s_branch .LBB8_104
.Lfunc_end8:
	.size	gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set, .Lfunc_end8-gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
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
		.amdhsa_next_free_vgpr 199
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end8-gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.num_vgpr, 199
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16280
; TotalNumSgprs: 40
; NumVgprs: 199
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 24
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 199
; Occupancy: 7
; WaveLimiterHint : 1
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
	.type	__hip_cuid_f88118b92911aa95,@object ; @__hip_cuid_f88118b92911aa95
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f88118b92911aa95
__hip_cuid_f88118b92911aa95:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f88118b92911aa95, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f88118b92911aa95
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
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         quantize_int4_mmq_ds128.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
    .name:           quantize_int4_mmq_ds128_atiled
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         quantize_int4_mmq_ds128_atiled.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4
    .private_segment_fixed_size: 0
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     202
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_full_add
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     202
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_full_set
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     202
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_atiled_add
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_atiled_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     199
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_atiled_set
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_atiled_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     199
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     199
    .vgpr_spill_count: 0
    .wavefront_size: 32
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
    .name:           gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_atiled_nt_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     199
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
