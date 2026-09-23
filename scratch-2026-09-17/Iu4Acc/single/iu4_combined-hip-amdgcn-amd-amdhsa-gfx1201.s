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
	v_max_num_f32_e32 v9, v9, v11
	v_xor_b32_e32 v11, 4, v6
	ds_bpermute_b32 v10, v8, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v6, v11, vcc_lo
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v10, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v9, v9, v12 :: v_dual_lshlrev_b32 v10, 2, v11
	v_xor_b32_e32 v12, 2, v6
	ds_bpermute_b32 v11, v10, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v12
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v12, v6, v12 :: v_dual_max_num_f32 v13, v11, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v12, v9, v13 :: v_dual_lshlrev_b32 v11, 2, v12
	v_xor_b32_e32 v13, 1, v6
	ds_bpermute_b32 v9, v11, v12
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v6, v13, vcc_lo
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v13, v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_max_num_f32 v6, v12, v13 :: v_dual_lshlrev_b32 v9, 2, v6
	ds_bpermute_b32 v12, v9, v6
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v12, v12, v12
	v_max_num_f32_e32 v12, v6, v12
	v_mov_b32_e32 v6, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v12
	s_cbranch_execz .LBB0_6
; %bb.4:                                ; %.preheader76.i
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
.LBB0_5:                                ; %.preheader.preheader.i
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
	v_cmp_lt_f32_e32 vcc_lo, v16, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v14, v16, vcc_lo
	v_cndmask_b32_e32 v6, v6, v15, vcc_lo
	s_cbranch_scc1 .LBB0_5
.LBB0_6:                                ; %Flow39
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_neq_f32_e64 s0, 0, v12
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v13, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_div_scale_f32 v13, null, v6, v6, v1
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v1, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v13, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v13, -v13, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v16
	v_div_fixup_f32 v1, v13, v6, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v13, v1
.LBB0_8:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_div_scale_f32 v1, null, v6, v6, v2
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v12, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	v_div_scale_f32 v14, vcc_lo, v2, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v12
	v_fma_f32 v16, -v1, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v16, v12
	v_fma_f32 v1, -v1, v15, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v12, v15
	v_div_fixup_f32 v1, v1, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v12, v1
.LBB0_10:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_div_scale_f32 v1, null, v6, v6, v3
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v1, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	v_div_scale_f32 v15, vcc_lo, v3, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v16, v15, v14
	v_fma_f32 v17, -v1, v16, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v14
	v_fma_f32 v1, -v1, v16, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v14, v16
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
	v_cvt_i32_f32_e32 v2, v2
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
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
	v_mad_co_i64_i32 v[3:4], null, v3, s3, 0
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
	v_add_co_u32 v7, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b16 v2.l, v2.h, v2.l
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	global_store_b16 v[7:8], v2, off offset:8
	s_and_b32 exec_lo, exec_lo, vcc_lo
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
; codeLenInByte = 2300
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
	s_lshl_b32 s11, ttmp9, 7
	v_mov_b32_e32 v68, v0
	s_lshl_b32 s22, ttmp7, 7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v69, 5, v68
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s11, s12
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s22, s14
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s23, s0, s1
	s_cmp_eq_u32 s15, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB1_15
; %bb.1:
	s_mov_b32 s24, 0
	s_and_b32 vcc_lo, exec_lo, s23
	s_cbranch_vccz .LBB1_16
; %bb.2:                                ; %.preheader508.i
	s_ashr_i32 s0, s13, 31
	s_add_co_i32 s2, s12, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s10, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v68
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v0, 1, v68
	v_and_b32_e32 v1, 1, v68
	v_mov_b32_e32 v3, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v4, s22, v0
	v_lshlrev_b32_e32 v2, 2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[3:4], null, 0x48, v4, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, v3, v2
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	global_load_b32 v3, v[3:4], off
.LBB1_5:                                ; %.preheader502.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, s11, v0
	v_lshlrev_b32_e32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v4, s2, v6
	v_cmp_gt_i32_e32 vcc_lo, s12, v6
	v_mad_co_i64_i32 v[4:5], null, s3, v4, s[16:17]
	global_load_b32 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_lshlrev_b32 v0, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, 0, v0, v2
	ds_store_2addr_stride64_b32 v0, v3, v1 offset0:48 offset1:56
.LBB1_6:                                ; %Flow1277
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v31, 2, v68
	v_dual_mov_b32 v5, 0 :: v_dual_and_b32 v0, 3, v68
	s_add_co_i32 s4, s14, -1
	v_lshlrev_b32_e32 v17, 4, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v32, s22, v31
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s11, v31
	v_lshlrev_b32_e32 v0, 3, v0
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v59, 64, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v2, 64, v1
	v_min_i32_e32 v3, s4, v32
	v_min_i32_e32 v1, s2, v1
	v_min_i32_e32 v4, s4, v59
	v_or_b32_e32 v49, 8, v69
	v_cmp_gt_i32_e64 s0, s14, v32
	v_cmp_gt_i32_e64 s1, s14, v59
	v_mad_co_u64_u32 v[231:232], null, 0x48, v3, v[0:1]
	v_mov_b32_e32 v3, 0
	v_min_i32_e32 v2, s2, v2
	v_mad_co_u64_u32 v[182:183], null, s3, v1, v[0:1]
	v_mad_co_u64_u32 v[183:184], null, 0x48, v4, v[0:1]
	v_mov_b32_e32 v4, 0
	v_dual_mov_b32 v19, 0 :: v_dual_and_b32 v50, 16, v17
	v_mad_co_u64_u32 v[184:185], null, s3, v2, v[0:1]
	global_load_b64 v[25:26], v231, s[18:19] offset:8
	global_load_b64 v[27:28], v182, s[16:17] offset:8
	global_load_b64 v[29:30], v183, s[18:19] offset:8
	global_load_b64 v[57:58], v184, s[16:17] offset:8
	v_bfe_u32 v0, v68, 1, 1
	v_and_or_b32 v31, v31, 15, v50
	v_mov_b32_e32 v56, 0
	v_bfe_u32 v70, v68, 4, 1
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v67, 15, v68
	v_and_or_b32 v50, v69, 6, v0
	v_lshlrev_b32_e32 v31, 3, v31
	v_and_or_b32 v0, v49, 14, v0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v60, v50, 8, v31
	v_lshl_or_b32 v0, v0, 8, v31
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v38, 0
	v_mov_b32_e32 v37, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v251, 0, v60
	v_add_nc_u32_e32 v252, 0, v0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v26, 0, v26, s0
	v_cndmask_b32_e64 v25, 0, v25, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v60, 0, v30, s1
	v_cndmask_b32_e64 v59, 0, v29, s1
	ds_store_2addr_stride64_b64 v251, v[25:26], v[27:28] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v252, v[59:60], v[57:58] offset1:8
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_18
; %bb.7:                                ; %.preheader501.lr.ph.i
	v_lshrrev_b32_e32 v0, 1, v68
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v2, 31, v68
	v_bfe_u32 v3, v68, 5, 1
	v_lshlrev_b32_e32 v7, 3, v67
	v_and_b32_e32 v9, 1, v68
	v_lshrrev_b32_e32 v4, 6, v68
	v_add_nc_u32_e32 v5, s11, v0
	v_lshlrev_b32_e32 v10, 3, v2
	v_lshl_or_b32 v11, v3, 9, v7
	v_add_nc_u32_e32 v7, s22, v0
	v_lshlrev_b32_e32 v0, 3, v0
	v_lshlrev_b32_e32 v12, 2, v9
	v_lshlrev_b32_e32 v2, 8, v4
	v_lshl_or_b32 v253, v4, 10, v10
	v_min_i32_e32 v4, s4, v7
	v_lshlrev_b32_e32 v6, 6, v70
	v_add3_u32 v0, 0, v0, v12
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s31, 0x1000
	s_clause 0x3                            ; 16-byte Folded Spill
	scratch_store_b32 off, v4, off offset:36
	scratch_store_b32 off, v70, off offset:60
	scratch_store_b32 off, v68, off offset:52
	scratch_store_b32 off, v0, off offset:40
	v_lshl_add_u32 v0, v3, 11, 0
	v_dual_mov_b32 v4, v1 :: v_dual_lshlrev_b32 v3, 4, v9
	v_lshlrev_b32_e32 v9, 2, v9
	s_movk_i32 s25, 0x3000
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v254, v0, v10
	v_mov_b32_e32 v0, v1
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v3, off offset:44
	scratch_store_b32 off, v67, off offset:64
	v_mov_b32_e32 v3, v1
	v_add3_u32 v2, 0, v2, v6
	v_dual_mov_b32 v6, v1 :: v_dual_add_nc_u32 v11, 0, v11
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:16
	scratch_store_b32 off, v0, off offset:20
	v_mov_b32_e32 v0, v231
	scratch_store_b32 off, v2, off offset:8 ; 4-byte Folded Spill
	v_mov_b32_e32 v2, v1
	v_min_i32_e32 v8, s2, v5
	v_mov_b32_e32 v65, v1
	s_movk_i32 s26, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	scratch_store_b64 off, v[0:1], off      ; 8-byte Folded Spill
	v_mad_co_u64_u32 v[13:14], null, s3, v8, s[16:17]
	v_ashrrev_i32_e32 v8, 31, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[14:15], null, s3, v8, v[14:15]
	v_mov_b32_e32 v8, v1
	v_cmp_gt_i32_e64 s2, s14, v7
	v_cmp_gt_i32_e64 s3, s12, v5
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v40, v8
	s_clause 0x1                            ; 16-byte Folded Spill
	scratch_store_b32 off, v11, off offset:12
	scratch_store_b96 off, v[13:15], off offset:24
	v_mov_b32_e32 v38, v6
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v69, off offset:56
	scratch_store_b32 off, v9, off offset:48
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v61, v5
	v_mov_b32_e32 v39, v7
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_mov_b32_e32 v33, v1
	v_dual_mov_b32 v35, v3 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v58, v2
	s_branch .LBB1_9
.LBB1_8:                                ;   in Loop: Header=BB1_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s28
	s_mov_b32 s6, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_9:                                ; %.preheader501.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_11 Depth 2
	s_mov_b32 s7, s5
	s_add_co_i32 s27, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s27, s10
	s_mov_b32 s29, -1
	s_cselect_b32 s28, -1, 0
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s30, s5
	s_mov_b32 s33, s5
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=2
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v178, v223, v179
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v173, v173
	v_fmac_f32_e32 v64, v178, v177
	v_mul_f32_e32 v177, v222, v179
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v165, v165
	v_fmac_f32_e32 v63, v177, v176
	v_mul_f32_e32 v176, v221, v179
	v_cvt_f32_i32_e32 v170, v170
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v166, v166
	v_fmac_f32_e32 v62, v176, v175
	v_mul_f32_e32 v175, v220, v179
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v163, v163
	v_fmac_f32_e32 v61, v175, v174
	v_mul_f32_e32 v174, v219, v179
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v159, v159
	v_fmac_f32_e32 v60, v174, v173
	v_mul_f32_e32 v173, v218, v179
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v156, v156
	v_fmac_f32_e32 v59, v173, v172
	v_mul_f32_e32 v172, v217, v179
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v150, v150
	v_fmac_f32_e32 v58, v172, v171
	v_mul_f32_e32 v171, v223, v255
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v147, v147
	v_fmac_f32_e32 v56, v171, v170
	v_mul_f32_e32 v170, v222, v255
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v143, v143
	v_fmac_f32_e32 v55, v170, v169
	v_mul_f32_e32 v169, v221, v255
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v140, v140
	v_fmac_f32_e32 v54, v169, v168
	v_mul_f32_e32 v168, v220, v255
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v135, v135
	v_fmac_f32_e32 v53, v168, v167
	v_mul_f32_e32 v167, v219, v255
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v127, v127
	v_fmac_f32_e32 v52, v167, v166
	v_mul_f32_e32 v166, v218, v255
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v118, v118
	v_fmac_f32_e32 v51, v166, v165
	v_mul_f32_e32 v165, v217, v255
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v116, v116
	v_fmac_f32_e32 v50, v165, v164
	v_mul_f32_e32 v164, v223, v248
	v_cvt_f32_i32_e32 v117, v117
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v113, v113
	v_dual_fmac_f32 v48, v164, v163 :: v_dual_mul_f32 v163, v222, v248
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v109, v109
	v_dual_fmac_f32 v47, v163, v162 :: v_dual_mul_f32 v162, v221, v248
	v_cvt_f32_i32_e32 v102, v102
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v111, v111
	v_dual_fmac_f32 v46, v162, v161 :: v_dual_mul_f32 v161, v220, v248
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v99, v99
	v_fmac_f32_e32 v45, v161, v160
	v_mul_f32_e32 v160, v219, v248
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v104, v104
	v_dual_fmac_f32 v44, v160, v159 :: v_dual_mul_f32 v159, v218, v248
	v_cvt_f32_i32_e32 v103, v103
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v88, v88
	v_cvt_f32_i32_e32 v107, v107
	v_dual_fmac_f32 v43, v159, v158 :: v_dual_mul_f32 v158, v217, v248
	v_cvt_f32_i32_e32 v94, v94
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v96, v96
	v_fmac_f32_e32 v42, v158, v157
	v_mul_f32_e32 v157, v223, v249
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v100, v100
	v_fmac_f32_e32 v40, v157, v156
	v_mul_f32_e32 v156, v222, v249
	v_cvt_f32_i32_e32 v83, v83
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v89, v89
	v_fmac_f32_e32 v39, v156, v155
	v_mul_f32_e32 v155, v221, v249
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cvt_f32_i32_e32 v74, v74
	v_cvt_f32_i32_e32 v91, v91
	v_fmac_f32_e32 v38, v155, v154
	v_mul_f32_e32 v154, v220, v249
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v76, v76
	v_fmac_f32_e32 v37, v154, v153
	v_mul_f32_e32 v153, v219, v249
	v_cvt_f32_i32_e32 v78, v78
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v69, v69
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v36, v153, v152
	v_mul_f32_e32 v152, v218, v249
	v_cvt_f32_i32_e32 v67, v67
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v86, v86
	v_cvt_f32_i32_e32 v73, v73
	v_fmac_f32_e32 v35, v152, v151
	v_mul_f32_e32 v151, v217, v249
	v_cvt_f32_i32_e32 v75, v75
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v72, v72
	v_fmac_f32_e32 v34, v151, v150
	v_mul_f32_e32 v150, v179, v231
	v_cvt_f32_i32_e32 v68, v68
	v_cvt_f32_i32_e32 v66, v66
	v_cvt_f32_i32_e32 v70, v70
	s_xor_b32 s4, s29, -1
	v_dual_fmac_f32 v32, v150, v149 :: v_dual_mul_f32 v149, v179, v230
	s_mov_b32 s33, 1
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s30, -1
	v_dual_fmac_f32 v31, v149, v148 :: v_dual_mul_f32 v148, v179, v227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v148, v147 :: v_dual_mul_f32 v147, v179, v226
	v_fmac_f32_e32 v29, v147, v146
	v_mul_f32_e32 v146, v179, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v146, v145 :: v_dual_mul_f32 v145, v179, v224
	v_dual_fmac_f32 v27, v145, v144 :: v_dual_mul_f32 v144, v179, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v144, v143
	v_mul_f32_e32 v143, v255, v231
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v142, v255, v230
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v142, v141
	v_mul_f32_e32 v141, v255, v227
	v_fmac_f32_e32 v22, v141, v140
	v_mul_f32_e32 v140, v255, v226
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v140, v139
	v_mul_f32_e32 v139, v255, v225
	v_fmac_f32_e32 v20, v139, v138
	v_mul_f32_e32 v138, v255, v224
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v138, v137
	v_mul_f32_e32 v137, v255, v229
	v_fmac_f32_e32 v18, v137, v136
	v_mul_f32_e32 v136, v248, v231
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v136, v135
	v_mul_f32_e32 v135, v248, v230
	v_dual_fmac_f32 v15, v135, v134 :: v_dual_mul_f32 v134, v248, v227
	v_mul_f32_e32 v135, v249, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v14, v134, v133 :: v_dual_mul_f32 v133, v248, v226
	v_mul_f32_e32 v134, v249, v224
	v_dual_fmac_f32 v13, v133, v132 :: v_dual_mul_f32 v132, v248, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v3, v134, v123
	v_mul_f32_e32 v133, v249, v227
	v_fmac_f32_e32 v12, v132, v131
	v_dual_mul_f32 v131, v248, v224 :: v_dual_mul_f32 v132, v249, v226
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v6, v133, v126
	v_dual_fmac_f32 v11, v131, v130 :: v_dual_mul_f32 v130, v248, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v131, v249, v231 :: v_dual_fmac_f32 v10, v130, v129
	v_mul_f32_e32 v129, v249, v229
	v_fmac_f32_e32 v2, v129, v122
	v_mul_f32_e32 v122, v239, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v236, v228
	v_dual_mul_f32 v130, v249, v230 :: v_dual_fmac_f32 v59, v115, v116
	v_mul_f32_e32 v115, v237, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v7, v130, v127 :: v_dual_fmac_f32 v60, v115, v117
	v_mul_f32_e32 v115, v234, v228
	v_fmac_f32_e32 v5, v132, v125
	v_fmac_f32_e32 v61, v115, v118
	v_mul_f32_e32 v115, v235, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v62, v115, v119 :: v_dual_mul_f32 v115, v232, v228
	v_fmac_f32_e32 v63, v115, v120
	v_mul_f32_e32 v115, v233, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v64, v115, v121
	v_mul_f32_e32 v115, v239, v0
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v236, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v8, v131, v128 :: v_dual_fmac_f32 v51, v108, v109
	v_mul_f32_e32 v108, v237, v0
	v_fmac_f32_e32 v52, v108, v110
	v_mul_f32_e32 v108, v234, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v135, v124 :: v_dual_fmac_f32 v53, v108, v111
	v_mul_f32_e32 v108, v235, v0
	v_fmac_f32_e32 v54, v108, v112
	v_mul_f32_e32 v108, v232, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v55, v108, v113 :: v_dual_mul_f32 v108, v233, v0
	v_fmac_f32_e32 v56, v108, v114
	v_mul_f32_e32 v108, v239, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v236, v180
	v_fmac_f32_e32 v43, v101, v102
	v_mul_f32_e32 v101, v237, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v44, v101, v103 :: v_dual_mul_f32 v101, v234, v180
	v_fmac_f32_e32 v45, v101, v104
	v_mul_f32_e32 v101, v235, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v46, v101, v105 :: v_dual_mul_f32 v101, v232, v180
	v_fmac_f32_e32 v47, v101, v106
	v_mul_f32_e32 v101, v233, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v48, v101, v107 :: v_dual_mul_f32 v101, v239, v181
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v236, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v35, v94, v95 :: v_dual_mul_f32 v94, v237, v181
	v_fmac_f32_e32 v36, v94, v96
	v_mul_f32_e32 v94, v234, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v94, v97
	v_mul_f32_e32 v94, v235, v181
	v_fmac_f32_e32 v38, v94, v98
	v_mul_f32_e32 v94, v232, v181
	scratch_load_b64 v[231:232], off, off   ; 8-byte Folded Reload
	v_dual_fmac_f32 v39, v94, v99 :: v_dual_mul_f32 v94, v233, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v40, v94, v100
	v_mul_f32_e32 v94, v228, v245
	v_dual_fmac_f32 v26, v94, v87 :: v_dual_mul_f32 v87, v228, v242
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v27, v87, v88
	v_mul_f32_e32 v87, v228, v243
	v_dual_fmac_f32 v28, v87, v89 :: v_dual_mul_f32 v87, v228, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v29, v87, v90
	v_mul_f32_e32 v87, v228, v241
	v_dual_fmac_f32 v30, v87, v91 :: v_dual_mul_f32 v87, v228, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v31, v87, v92
	v_mul_f32_e32 v87, v228, v247
	v_fmac_f32_e32 v32, v87, v93
	v_mul_f32_e32 v87, v0, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v87, v80
	v_mul_f32_e32 v80, v0, v242
	v_fmac_f32_e32 v19, v80, v81
	v_mul_f32_e32 v80, v0, v243
	v_cvt_f32_i32_e32 v81, v190
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v20, v80, v82
	v_mul_f32_e32 v80, v0, v240
	v_fmac_f32_e32 v21, v80, v83
	v_mul_f32_e32 v80, v0, v241
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v80, v84
	v_mul_f32_e32 v80, v0, v246
	v_mul_f32_e32 v0, v0, v247
	v_fmac_f32_e32 v24, v0, v86
	v_mul_f32_e32 v0, v180, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v10, v0, v73
	v_mul_f32_e32 v0, v180, v242
	v_mul_f32_e32 v73, v181, v246
	v_fmac_f32_e32 v7, v73, v71
	v_mul_f32_e32 v73, v215, v192
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v11, v0, v74
	v_dual_mul_f32 v0, v180, v243 :: v_dual_mul_f32 v71, v215, v194
	v_fmac_f32_e32 v12, v0, v75
	v_dual_mul_f32 v0, v180, v240 :: v_dual_mul_f32 v75, v181, v242
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v13, v0, v76
	v_dual_mul_f32 v0, v180, v241 :: v_dual_fmac_f32 v3, v75, v67
	v_mul_f32_e32 v67, v215, v198
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v75, v189, v200 :: v_dual_fmac_f32 v14, v0, v77
	v_dual_mul_f32 v0, v180, v246 :: v_dual_mul_f32 v77, v181, v240
	v_fmac_f32_e32 v15, v0, v78
	v_mul_f32_e32 v0, v180, v247
	v_mul_f32_e32 v78, v181, v241
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v5, v77, v69
	v_mul_f32_e32 v69, v215, v200
	v_dual_mul_f32 v77, v189, v194 :: v_dual_fmac_f32 v16, v0, v79
	v_mul_f32_e32 v0, v181, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v79, v189, v191 :: v_dual_fmac_f32 v8, v0, v72
	v_cvt_f32_i32_e32 v0, v216
	v_mul_f32_e32 v76, v181, v243
	v_mul_f32_e32 v72, v215, v193
	v_mul_f32_e32 v74, v181, v245
	v_dual_fmac_f32 v3, v79, v81 :: v_dual_fmac_f32 v32, v69, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v4, v76, v68
	v_dual_mul_f32 v68, v215, v191 :: v_dual_fmac_f32 v29, v72, v0
	v_mul_f32_e32 v72, v213, v193
	v_fmac_f32_e32 v2, v74, v66
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v66, v210, v215 :: v_dual_fmac_f32 v27, v68, v0
	v_mul_f32_e32 v68, v213, v191
	v_fmac_f32_e32 v6, v78, v70
	v_mul_f32_e32 v70, v215, v199
	v_mul_f32_e32 v78, v189, v193
	v_fmac_f32_e32 v64, v66, v0
	v_fmac_f32_e32 v28, v73, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v26, v67, v0 :: v_dual_fmac_f32 v31, v70, v0
	v_dual_mul_f32 v70, v213, v199 :: v_dual_fmac_f32 v5, v78, v81
	v_mul_f32_e32 v66, v209, v215
	v_mul_f32_e32 v76, v189, v199
	v_mul_f32_e32 v67, v213, v198
	v_mul_f32_e32 v69, v213, v200
	v_mul_f32_e32 v73, v213, v192
	v_fmac_f32_e32 v63, v66, v0
	v_mul_f32_e32 v66, v206, v215
	v_dual_mul_f32 v74, v189, v198 :: v_dual_fmac_f32 v7, v76, v81
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v62, v66, v0
	v_mul_f32_e32 v66, v205, v215
	v_dual_fmac_f32 v61, v66, v0 :: v_dual_mul_f32 v66, v204, v215
	v_dual_fmac_f32 v23, v80, v85 :: v_dual_fmac_f32 v30, v71, v0
	v_mul_f32_e32 v71, v213, v194
	v_mul_f32_e32 v80, v189, v192
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v60, v66, v0
	v_mul_f32_e32 v66, v203, v215
	v_dual_fmac_f32 v59, v66, v0 :: v_dual_mul_f32 v66, v208, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v58, v66, v0
	v_mul_f32_e32 v66, v207, v215
	v_dual_fmac_f32 v57, v66, v0 :: v_dual_mul_f32 v66, v215, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v66, v0
	v_cvt_f32_i32_e32 v0, v214
	v_mul_f32_e32 v66, v210, v213
	v_fmac_f32_e32 v24, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v66, v0
	v_dual_mul_f32 v66, v209, v213 :: v_dual_fmac_f32 v23, v70, v0
	v_dual_fmac_f32 v22, v71, v0 :: v_dual_fmac_f32 v21, v72, v0
	v_dual_fmac_f32 v20, v73, v0 :: v_dual_fmac_f32 v55, v66, v0
	v_dual_mul_f32 v66, v206, v213 :: v_dual_fmac_f32 v19, v68, v0
	v_fmac_f32_e32 v18, v67, v0
	v_dual_mul_f32 v67, v211, v198 :: v_dual_mul_f32 v68, v211, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v54, v66, v0
	v_dual_mul_f32 v66, v205, v213 :: v_dual_mul_f32 v69, v211, v200
	v_dual_mul_f32 v70, v211, v199 :: v_dual_mul_f32 v71, v211, v194
	v_dual_mul_f32 v72, v211, v193 :: v_dual_fmac_f32 v53, v66, v0
	v_dual_mul_f32 v66, v204, v213 :: v_dual_mul_f32 v73, v211, v192
	v_fmac_f32_e32 v6, v77, v81
	v_fmac_f32_e32 v4, v80, v81
	v_fmac_f32_e32 v2, v74, v81
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v66, v0
	v_mul_f32_e32 v66, v203, v213
	v_dual_fmac_f32 v51, v66, v0 :: v_dual_mul_f32 v66, v208, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v50, v66, v0
	v_mul_f32_e32 v66, v207, v213
	v_dual_fmac_f32 v49, v66, v0 :: v_dual_mul_f32 v66, v213, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v66, v0
	v_cvt_f32_i32_e32 v0, v212
	v_mul_f32_e32 v66, v210, v211
	v_fmac_f32_e32 v16, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v48, v66, v0
	v_dual_mul_f32 v66, v209, v211 :: v_dual_fmac_f32 v15, v70, v0
	v_dual_fmac_f32 v14, v71, v0 :: v_dual_fmac_f32 v13, v72, v0
	v_dual_fmac_f32 v12, v73, v0 :: v_dual_fmac_f32 v47, v66, v0
	v_dual_mul_f32 v66, v206, v211 :: v_dual_fmac_f32 v11, v68, v0
	v_dual_fmac_f32 v10, v67, v0 :: v_dual_mul_f32 v67, v210, v189
	v_mul_f32_e32 v68, v209, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v46, v66, v0
	v_dual_mul_f32 v66, v205, v211 :: v_dual_mul_f32 v69, v206, v189
	v_dual_mul_f32 v70, v205, v189 :: v_dual_mul_f32 v71, v203, v189
	v_dual_mul_f32 v72, v204, v189 :: v_dual_fmac_f32 v45, v66, v0
	v_dual_mul_f32 v66, v204, v211 :: v_dual_mul_f32 v73, v189, v197
	v_dual_fmac_f32 v40, v67, v81 :: v_dual_fmac_f32 v39, v68, v81
	v_fmac_f32_e32 v38, v69, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v66, v0
	v_dual_mul_f32 v66, v203, v211 :: v_dual_fmac_f32 v37, v70, v81
	v_dual_fmac_f32 v36, v72, v81 :: v_dual_fmac_f32 v35, v71, v81
	v_dual_fmac_f32 v8, v75, v81 :: v_dual_fmac_f32 v43, v66, v0
	v_dual_mul_f32 v66, v208, v211 :: v_dual_fmac_f32 v1, v73, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v42, v66, v0
	v_mul_f32_e32 v66, v207, v211
	v_dual_fmac_f32 v41, v66, v0 :: v_dual_mul_f32 v66, v211, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v9, v66, v0 :: v_dual_mul_f32 v0, v207, v189
	v_mul_f32_e32 v66, v208, v189
	v_dual_fmac_f32 v33, v0, v81 :: v_dual_fmac_f32 v34, v66, v81
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_8
.LBB1_11:                               ;   Parent Loop BB1_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	scratch_load_b32 v0, off, off offset:8  ; 4-byte Folded Reload
	s_or_b32 s4, s33, s7
	s_and_b32 s34, s29, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_cselect_b32 s38, s25, 0x3400
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_cselect_b32 s39, s26, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	s_mov_b32 s37, s5
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v66, s36, s34, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, s35, 0, s36
	s_lshl_b32 s36, s33, 6
	v_add_co_u32 v68, s34, s34, v183
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[8:9], s[36:37]
	v_add_co_ci_u32_e64 v69, null, s35, 0, s34
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v70, s34, s36, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v71, null, s37, 0, s34
	v_add_co_u32 v72, s34, s36, v184
	s_clause 0x1
	global_load_b64 v[66:67], v[66:67], off offset:40
	global_load_b64 v[68:69], v[68:69], off offset:40
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s37, 0, s34
	s_clause 0x1
	global_load_b64 v[185:186], v[70:71], off offset:40
	global_load_b64 v[187:188], v[72:73], off offset:40
	v_add_nc_u32_e32 v76, s31, v253
	ds_load_b64 v[70:71], v76
	s_wait_dscnt 0x0
	v_dual_mov_b32 v75, v71 :: v_dual_mov_b32 v74, v70
	s_wait_loadcnt 0x4
	v_add_nc_u32_e32 v250, s39, v0
	ds_load_b64 v[72:73], v254
	ds_load_2addr_b32 v[215:216], v250 offset1:2
	ds_load_2addr_b32 v[217:218], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[219:220], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[221:222], v250 offset0:12 offset1:14
	;;#ASMSTART
	v_xor_b32 v74, v74, v65
	;;#ASMEND
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[170:177], v[74:75], v[72:73], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v196, 0, v67, s0
	v_cndmask_b32_e64 v195, 0, v66, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v202, 0, v69, s1
	v_cndmask_b32_e64 v201, 0, v68, s1
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v0, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f32_i32_e32 v65, v170
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v178, s38, v0
	ds_load_b32 v226, v178
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v226
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v57, v0, v65, v57
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v70 :: v_dual_mov_b32 v68, v71
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v225, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v225
	v_fma_f32 v49, v0, v65, v49
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v70
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v224, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v224
	v_fma_f32 v41, v0, v65, v41
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v70, v70, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[70:71], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v223, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v223
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v0, v65, v33
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:512
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[208:209], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[210:211], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[212:213], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[214:215], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v226, v208
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v0, v67, v25
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v225, v208
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v17, v0, v67, v17
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v224, v208
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, v0, v67, v9
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v223, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fma_f32 v1, v0, v65, v1
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:256
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[207:208], v250 offset1:2
	ds_load_2addr_b32 v[205:206], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[203:204], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[199:200], v250 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v181
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:128
	v_cvt_f32_i32_e32 v67, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v180
	v_fmac_f32_e32 v49, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178 offset:256
	v_cvt_f32_i32_e32 v67, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v179
	v_fmac_f32_e32 v41, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v207, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[227:228], v76 offset:768
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[197:198], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[193:194], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[191:192], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[189:190], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v227 :: v_dual_mov_b32 v68, v228
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v181, v197
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v197
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v179, v197
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[229:230], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v227, v227, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[227:228], v[229:230], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v197, v0, v197
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v197, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s31, s30, s28
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s31
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:16
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v195, off, off offset:36
	scratch_load_b32 v197, off, off offset:48
	s_add_co_i32 s4, s4, 1
	s_mov_b32 s39, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_add_co_i32 s4, s33, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[4:5], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	s_xor_b32 s33, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v185, s38, s34, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v186, null, s35, 0, s38
	s_add_nc_u64 s[36:37], s[16:17], s[36:37]
	s_lshl_b32 s38, s33, 6
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[36:37], s[38:39]
	v_add_co_u32 v187, s40, s34, v183
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v188, null, s35, 0, s40
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[195:196], null, 0x48, v195, s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v227, s34, s36, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v228, null, s37, 0, s34
	v_add_co_u32 v229, s34, s36, v184
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v230, null, s37, 0, s34
	s_wait_loadcnt 0x0
	v_add_co_u32 v231, vcc_lo, v195, v197
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v232, null, 0, v196, vcc_lo
	scratch_load_b96 v[195:197], off, off offset:24 ; 12-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_co_u32 v195, vcc_lo, v195, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v196, null, 0, v196, vcc_lo
	s_lshl_b32 s4, s33, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v233, vcc_lo, v195, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v234, null, 0, v196, vcc_lo
	s_clause 0x1
	global_load_b64 v[195:196], v[185:186], off offset:8
	global_load_b64 v[201:202], v[187:188], off offset:8
	s_clause 0x1
	global_load_b64 v[185:186], v[227:228], off offset:8
	global_load_b64 v[187:188], v[229:230], off offset:8
	global_load_b32 v197, v[231:232], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:16 ; 4-byte Folded Spill
	global_load_b32 v197, v[233:234], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:20 ; 4-byte Folded Spill
.LBB1_13:                               ; %.preheader499.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	v_dual_mul_f32 v197, v222, v226 :: v_dual_mul_f32 v228, v219, v226
	v_mul_f32_e32 v207, v221, v226
	v_dual_mul_f32 v227, v220, v226 :: v_dual_mul_f32 v230, v217, v226
	v_mul_f32_e32 v231, v216, v226
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v229, v218, v226
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_fmac_f32_e32 v64, v197, v177
	v_fma_f32 v59, v230, v172, v59
	v_fma_f32 v60, v229, v173, v60
	v_fma_f32 v61, v228, v174, v61
	v_fma_f32 v62, v227, v175, v62
	v_mul_f32_e32 v172, v221, v225
	v_fma_f32 v63, v207, v176, v63
	v_mul_f32_e32 v174, v219, v225
	v_fma_f32 v58, v231, v171, v58
	v_dual_mul_f32 v171, v222, v225 :: v_dual_mul_f32 v176, v217, v225
	v_mul_f32_e32 v175, v218, v225
	v_mul_f32_e32 v177, v216, v225
	v_cvt_f32_i32_e32 v170, v170
	v_mul_f32_e32 v173, v220, v225
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_fma_f32 v50, v177, v164, v50
	v_fma_f32 v51, v176, v165, v51
	v_fma_f32 v52, v175, v166, v52
	v_fma_f32 v53, v174, v167, v53
	v_fma_f32 v55, v172, v169, v55
	v_fmac_f32_e32 v56, v171, v170
	v_fma_f32 v54, v173, v168, v54
	v_dual_mul_f32 v164, v222, v224 :: v_dual_mul_f32 v165, v221, v224
	v_mul_f32_e32 v168, v218, v224
	v_dual_mul_f32 v166, v220, v224 :: v_dual_mul_f32 v167, v219, v224
	v_mul_f32_e32 v170, v216, v224
	v_cvt_f32_i32_e32 v163, v163
	v_mul_f32_e32 v169, v217, v224
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v162, v162
	v_fma_f32 v42, v170, v157, v42
	v_fma_f32 v44, v168, v159, v44
	v_fma_f32 v45, v167, v160, v45
	v_fma_f32 v46, v166, v161, v46
	v_fmac_f32_e32 v48, v164, v163
	v_fma_f32 v43, v169, v158, v43
	v_dual_mul_f32 v157, v222, v223 :: v_dual_mul_f32 v158, v221, v223
	v_dual_mul_f32 v159, v220, v223 :: v_dual_mul_f32 v160, v219, v223
	v_mul_f32_e32 v161, v218, v223
	v_cvt_f32_i32_e32 v156, v156
	v_mul_f32_e32 v163, v216, v223
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v153, v153
	v_fma_f32 v47, v165, v162, v47
	v_mul_f32_e32 v162, v217, v223
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v154, v154
	v_fma_f32 v36, v161, v152, v36
	v_fma_f32 v37, v160, v153, v37
	v_fmac_f32_e32 v40, v157, v156
	v_fma_f32 v34, v163, v150, v34
	v_mul_f32_e32 v150, v226, v215
	v_dual_mul_f32 v152, v226, v213 :: v_dual_mul_f32 v153, v226, v212
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v146, v146
	v_fma_f32 v35, v162, v151, v35
	v_fma_f32 v38, v159, v154, v38
	v_fma_f32 v39, v158, v155, v39
	v_dual_mul_f32 v151, v226, v214 :: v_dual_mul_f32 v154, v226, v211
	v_mul_f32_e32 v155, v226, v210
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v147, v147
	v_fma_f32 v29, v153, v146, v29
	v_fmac_f32_e32 v32, v150, v149
	v_dual_mul_f32 v146, v225, v212 :: v_dual_mul_f32 v149, v225, v209
	v_cvt_f32_i32_e32 v136, v136
	v_mul_f32_e32 v156, v226, v209
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v145, v145
	v_fma_f32 v27, v155, v144, v27
	v_fma_f32 v30, v152, v147, v30
	v_fma_f32 v31, v151, v148, v31
	v_dual_mul_f32 v144, v225, v214 :: v_dual_mul_f32 v147, v225, v211
	v_mul_f32_e32 v148, v225, v210
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_fma_f32 v18, v149, v136, v18
	v_mul_f32_e32 v136, v224, v215
	v_cvt_f32_i32_e32 v135, v135
	v_fma_f32 v26, v156, v143, v26
	v_fma_f32 v28, v154, v145, v28
	v_mul_f32_e32 v143, v225, v215
	v_mul_f32_e32 v145, v225, v213
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v140, v140
	v_fma_f32 v20, v147, v138, v20
	v_fma_f32 v21, v146, v139, v21
	v_fma_f32 v23, v144, v141, v23
	v_dual_mul_f32 v138, v224, v213 :: v_dual_mul_f32 v141, v224, v210
	v_mul_f32_e32 v139, v224, v212
	v_fma_f32 v19, v148, v137, v19
	v_mul_f32_e32 v137, v224, v214
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v130, v130
	v_dual_fmac_f32 v16, v136, v135 :: v_dual_mul_f32 v135, v223, v209
	v_cvt_f32_i32_e32 v122, v122
	v_fma_f32 v22, v145, v140, v22
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v140, v224, v211
	v_mul_f32_e32 v142, v224, v209
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_fma_f32 v11, v141, v130, v11
	v_fma_f32 v15, v137, v134, v15
	v_mul_f32_e32 v130, v223, v214
	v_mul_f32_e32 v134, v223, v210
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v123, v123
	v_fma_f32 v2, v135, v122, v2
	v_mul_f32_e32 v122, v208, v181
	v_cvt_f32_i32_e32 v115, v115
	v_fma_f32 v10, v142, v129, v10
	v_fma_f32 v12, v140, v131, v12
	v_fma_f32 v13, v139, v132, v13
	v_fma_f32 v14, v138, v133, v14
	v_mul_f32_e32 v129, v223, v215
	v_mul_f32_e32 v131, v223, v213
	v_mul_f32_e32 v133, v223, v211
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v126, v126
	v_fma_f32 v7, v130, v127, v7
	v_mul_f32_e32 v127, v199, v181
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v118, v118
	v_mul_f32_e32 v132, v223, v212
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v208, v180
	v_fma_f32 v3, v134, v123, v3
	v_mul_f32_e32 v123, v205, v181
	v_fma_f32 v4, v133, v124, v4
	v_fma_f32 v5, v132, v125, v5
	v_fma_f32 v6, v131, v126, v6
	v_dual_fmac_f32 v8, v129, v128 :: v_dual_mul_f32 v125, v203, v181
	v_dual_mul_f32 v124, v206, v181 :: v_dual_fmac_f32 v63, v127, v120
	v_dual_mul_f32 v126, v204, v181 :: v_dual_fmac_f32 v59, v123, v116
	v_mul_f32_e32 v128, v200, v181
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v117, v117
	v_mul_f32_e32 v120, v199, v180
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v113, v113
	v_mul_f32_e32 v116, v205, v180
	v_dual_fmac_f32 v62, v126, v119 :: v_dual_mul_f32 v119, v204, v180
	v_dual_fmac_f32 v61, v125, v118 :: v_dual_fmac_f32 v60, v124, v117
	v_mul_f32_e32 v117, v206, v180
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v111, v111
	v_dual_fmac_f32 v55, v120, v113 :: v_dual_mul_f32 v118, v203, v180
	v_fmac_f32_e32 v51, v116, v109
	v_fmac_f32_e32 v64, v128, v121
	v_mul_f32_e32 v121, v200, v180
	v_mul_f32_e32 v113, v199, v179
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v110, v110
	v_fmac_f32_e32 v56, v121, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v119, v112 :: v_dual_fmac_f32 v47, v113, v106
	v_fmac_f32_e32 v53, v118, v111
	v_dual_mul_f32 v111, v203, v179 :: v_dual_mul_f32 v112, v204, v179
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v104, v104
	v_mul_f32_e32 v106, v199, v0
	v_cvt_f32_i32_e32 v99, v99
	v_fmac_f32_e32 v52, v117, v110
	v_dual_mul_f32 v109, v205, v179 :: v_dual_mul_f32 v110, v206, v179
	v_mul_f32_e32 v114, v200, v179
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v208, v179
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v103, v103
	v_dual_fmac_f32 v46, v112, v105 :: v_dual_fmac_f32 v39, v106, v99
	v_fmac_f32_e32 v45, v111, v104
	v_dual_mul_f32 v104, v203, v0 :: v_dual_mul_f32 v105, v204, v0
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v97, v97
	v_mul_f32_e32 v99, v181, v189
	v_cvt_f32_i32_e32 v92, v92
	v_fmac_f32_e32 v48, v114, v107
	v_dual_fmac_f32 v44, v110, v103 :: v_dual_fmac_f32 v43, v109, v102
	v_dual_mul_f32 v102, v205, v0 :: v_dual_mul_f32 v103, v206, v0
	v_mul_f32_e32 v107, v200, v0
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v95, v95
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v208, v0
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v96, v96
	v_dual_fmac_f32 v38, v105, v98 :: v_dual_fmac_f32 v31, v99, v92
	v_fmac_f32_e32 v37, v104, v97
	v_mul_f32_e32 v97, v181, v191
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v74, v74
	v_mul_f32_e32 v92, v180, v189
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v107, v100 :: v_dual_fmac_f32 v29, v97, v90
	v_fmac_f32_e32 v36, v103, v96
	v_dual_mul_f32 v96, v181, v194 :: v_dual_fmac_f32 v23, v92, v85
	v_mul_f32_e32 v98, v181, v192
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v88, v88
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v181, v198
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v91, v91
	v_dual_mul_f32 v90, v180, v191 :: v_dual_mul_f32 v85, v179, v189
	v_cvt_f32_i32_e32 v83, v83
	v_mul_f32_e32 v100, v181, v190
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v21, v90, v83 :: v_dual_fmac_f32 v32, v100, v93
	v_mul_f32_e32 v83, v179, v191
	v_dual_mul_f32 v93, v180, v190 :: v_dual_fmac_f32 v30, v98, v91
	v_dual_mul_f32 v91, v180, v192 :: v_dual_fmac_f32 v26, v94, v87
	v_mul_f32_e32 v87, v180, v198
	v_fmac_f32_e32 v35, v102, v95
	v_mul_f32_e32 v95, v181, v193
	v_cvt_f32_i32_e32 v78, v78
	v_fmac_f32_e32 v28, v96, v89
	v_mul_f32_e32 v89, v180, v194
	v_cvt_f32_i32_e32 v76, v76
	v_dual_fmac_f32 v27, v95, v88 :: v_dual_mul_f32 v88, v180, v193
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v86, v86
	v_fmac_f32_e32 v20, v89, v82
	v_mul_f32_e32 v82, v179, v194
	v_cvt_f32_i32_e32 v75, v75
	v_fmac_f32_e32 v13, v83, v76
	v_fmac_f32_e32 v19, v88, v81
	v_dual_mul_f32 v81, v179, v193 :: v_dual_fmac_f32 v24, v93, v86
	v_fmac_f32_e32 v18, v87, v80
	v_cvt_f32_i32_e32 v73, v73
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v12, v82, v75 :: v_dual_fmac_f32 v11, v81, v74
	v_add_nc_u32_e32 v75, 0, v253
	v_mul_f32_e32 v81, v0, v189
	v_cvt_f32_i32_e32 v71, v71
	v_mul_f32_e32 v80, v179, v198
	v_cvt_f32_i32_e32 v82, v66
	v_cvt_f32_i32_e32 v83, v67
	ds_load_b64 v[66:67], v254
	v_dual_fmac_f32 v7, v81, v71 :: v_dual_fmac_f32 v10, v80, v73
	ds_load_b64 v[73:74], v75 offset:8192
	ds_load_2addr_b32 v[216:217], v250 offset1:2
	ds_load_2addr_b32 v[218:219], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[220:221], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[222:223], v250 offset0:12 offset1:14
	v_cvt_f32_i32_e32 v84, v84
	v_mul_f32_e32 v86, v179, v190
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v69, v69
	v_fmac_f32_e32 v22, v91, v84
	v_mul_f32_e32 v84, v179, v192
	v_fmac_f32_e32 v16, v86, v79
	v_mul_f32_e32 v79, v0, v191
	v_fmac_f32_e32 v15, v85, v78
	v_mul_f32_e32 v76, v0, v198
	v_fmac_f32_e32 v14, v84, v77
	v_dual_mul_f32 v77, v0, v193 :: v_dual_mul_f32 v78, v0, v194
	v_mul_f32_e32 v80, v0, v192
	v_mul_f32_e32 v0, v0, v190
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v70, v70
	v_cvt_f32_i32_e32 v84, v68
	v_fmac_f32_e32 v5, v79, v69
	s_wait_dscnt 0x4
	v_dual_mov_b32 v69, v74 :: v_dual_mov_b32 v68, v73
	v_fmac_f32_e32 v8, v0, v72
	v_fmac_f32_e32 v6, v80, v70
	v_dual_fmac_f32 v4, v78, v84 :: v_dual_fmac_f32 v3, v77, v83
	v_fmac_f32_e32 v2, v76, v82
	;;#ASMSTART
	v_xor_b32 v68, v68, v65
	;;#ASMEND
	v_wmma_i32_16x16x32_iu4 v[170:177], v[68:69], v[66:67], 0 neg_lo:[0,1,0]
	s_xor_b32 s4, s31, -1
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178
	v_cvt_f32_i32_e32 v65, v170
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v179
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v57, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v73 :: v_dual_mov_b32 v68, v74
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v255, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v255
	v_fmac_f32_e32 v49, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v73
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v248, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v248
	v_fmac_f32_e32 v41, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v73, v73, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[73:74], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v249, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v249
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8704
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[228:229], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[224:225], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[226:227], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[230:231], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v179, v228
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v255, v228
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v248, v228
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v249, v228
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fmac_f32_e32 v1, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8448
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[238:239], v250 offset1:2
	ds_load_2addr_b32 v[236:237], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[234:235], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[232:233], v250 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v228, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v238, v228
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:128
	v_cvt_f32_i32_e32 v68, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v0
	v_fmac_f32_e32 v49, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:256
	v_cvt_f32_i32_e32 v68, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v180
	v_fmac_f32_e32 v41, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v238, v181
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[189:190], v75 offset:8960
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[244:245], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[242:243], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[240:241], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[246:247], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v189 :: v_dual_mov_b32 v68, v190
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v228, v244
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v0, v244
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v244
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[191:192], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v189, v189, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[189:190], v[191:192], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v189, v181, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v189, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b32 v[207:208], v250 offset0:1 offset1:3
	ds_load_2addr_b32 v[203:204], v250 offset0:5 offset1:7
	ds_load_2addr_b32 v[205:206], v250 offset0:9 offset1:11
	ds_load_2addr_b32 v[209:210], v250 offset0:13 offset1:15
	ds_load_2addr_b32 v[215:216], v178 offset1:1
	ds_load_2addr_b32 v[213:214], v178 offset0:32 offset1:33
	ds_load_2addr_b32 v[211:212], v178 offset0:64 offset1:65
	ds_load_2addr_b32 v[189:190], v178 offset0:96 offset1:97
	ds_load_2addr_b32 v[197:198], v250 offset0:33 offset1:35
	ds_load_2addr_b32 v[191:192], v250 offset0:37 offset1:39
	ds_load_2addr_b32 v[193:194], v250 offset0:41 offset1:43
	ds_load_2addr_b32 v[199:200], v250 offset0:45 offset1:47
	s_movk_i32 s31, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
; %bb.14:                               ; %.preheader500.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v178, off, off offset:44
	scratch_load_b32 v238, off, off offset:20
	s_and_b32 s4, s30, exec_lo
	s_cselect_b32 s4, s25, 0x3400
	v_cndmask_b32_e64 v196, 0, v196, s0
	v_cndmask_b32_e64 v195, 0, v195, s0
	v_cndmask_b32_e64 v202, 0, v202, s1
	v_cndmask_b32_e64 v201, 0, v201, s1
	s_movk_i32 s31, 0x1000
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:8
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:8
	scratch_load_b32 v250, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v178, v238
	scratch_load_b32 v238, off, off offset:16 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v178, v178.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v244, s4, v250
	s_cselect_b32 s4, s26, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v250, s4, v250
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v238, 0, v238, s2
	ds_store_b32 v244, v238
	ds_store_b32 v250, v178
	s_branch .LBB1_10
.LBB1_15:
	s_mov_b32 s24, -1
.LBB1_16:                               ; %Flow1282
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s24
	s_cbranch_vccnz .LBB1_108
	s_branch .LBB1_252
.LBB1_17:                               ; %.preheader497.i.loopexit
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v68, off, off offset:52
	scratch_load_b32 v69, off, off offset:56
	scratch_load_b32 v70, off, off offset:60
	scratch_load_b32 v67, off, off offset:64
.LBB1_18:                               ; %.preheader497.i
	s_wait_loadcnt 0x2
	v_mul_u32_u24_e32 v0, 0x500, v69
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v65, 2, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v65, 0, v0, v65
	v_lshrrev_b32_e32 v0, 4, v68
	v_mad_u32_u24 v66, 0x280, v70, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v0, 15, v0
	ds_store_2addr_b32 v66, v57, v58 offset1:20
	ds_store_2addr_b32 v66, v59, v60 offset0:40 offset1:60
	ds_store_2addr_b32 v66, v61, v62 offset0:80 offset1:100
	ds_store_2addr_b32 v66, v63, v64 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v57, s11, v67
	v_or_b32_e32 v60, s22, v0
	v_mul_u32_u24_e32 v59, 0x50, v67
	v_lshlrev_b32_e32 v0, 2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v57
	v_cmp_gt_i32_e32 vcc_lo, s14, v60
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v0, 0, v59, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_20
; %bb.19:
	v_mad_co_i64_i32 v[61:62], null, s12, v60, 0
	v_lshlrev_b64_e32 v[63:64], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	v_add_co_u32 v59, s0, s20, v61
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, s21, v62, s0
	v_add_co_u32 v61, s0, v59, v63
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, v62, v64, s0
	ds_load_b32 v63, v0
	global_load_b32 v59, v[61:62], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v63, v59
	global_store_b32 v[61:62], v59, off
.LBB1_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v61, 64, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v61
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_22
; %bb.21:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:1280
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off
.LBB1_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 32, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v59
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_24
; %bb.23:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:2560
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:128
.LBB1_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_26
; %bb.25:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:3840
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:128
.LBB1_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 64, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v59
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_28
; %bb.27:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:5120
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:256
.LBB1_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_30
; %bb.29:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:6400
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:256
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 0x60, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v59
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_32
; %bb.31:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:7680
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:384
.LBB1_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v59, 0x280, v70
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_34
; %bb.33:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	ds_load_b32 v66, v0 offset:8960
	global_load_b32 v64, v[62:63], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v64, v66, v64
	global_store_b32 v[62:63], v64, off offset:384
.LBB1_34:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v59, v65, v59
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v49, v50 offset1:20
	ds_store_2addr_b32 v59, v51, v52 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v53, v54 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v55, v56 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v49, 16, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v49
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_36
; %bb.35:
	v_mad_co_i64_i32 v[50:51], null, s12, v49, 0
	v_lshlrev_b64_e32 v[52:53], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v50, s2, s20, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, s21, v51, s2
	v_add_co_u32 v50, s2, v50, v52
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, v51, v53, s2
	ds_load_b32 v53, v0
	global_load_b32 v52, v[50:51], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v52, v53, v52
	global_store_b32 v[50:51], v52, off
.LBB1_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v50, 0x50, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v50
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_122
; %bb.37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_123
.LBB1_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_124
.LBB1_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_125
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_126
.LBB1_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_127
.LBB1_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_44
.LBB1_43:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:8960
	global_load_b32 v53, v[51:52], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:384
.LBB1_44:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v41, v42 offset1:20
	ds_store_2addr_b32 v59, v43, v44 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v45, v46 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v47, v48 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v41
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_46
; %bb.45:
	v_mad_co_i64_i32 v[42:43], null, s12, v41, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v42, s4, s20, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s21, v43, s4
	v_add_co_u32 v42, s4, v42, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, v43, v45, s4
	ds_load_b32 v45, v0
	global_load_b32 v44, v[42:43], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v44, v45, v44
	global_store_b32 v[42:43], v44, off
.LBB1_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v42, 0x60, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v42
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_128
; %bb.47:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_129
.LBB1_48:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_130
.LBB1_49:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_131
.LBB1_50:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_132
.LBB1_51:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_133
.LBB1_52:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_54
.LBB1_53:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:8960
	global_load_b32 v45, v[43:44], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:384
.LBB1_54:                               ; %.preheader.3.i
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v33, v34 offset1:20
	ds_store_2addr_b32 v59, v35, v36 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v37, v38 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v39, v40 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v33, 48, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v33
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s15, s6
	s_cbranch_execz .LBB1_56
; %bb.55:
	v_mad_co_i64_i32 v[34:35], null, s12, v33, 0
	v_lshlrev_b64_e32 v[36:37], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v34, s6, s20, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, s21, v35, s6
	v_add_co_u32 v34, s6, v34, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, v35, v37, s6
	ds_load_b32 v37, v0
	global_load_b32 v36, v[34:35], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v36, v37, v36
	global_store_b32 v[34:35], v36, off
.LBB1_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v34, 0x70, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v34
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_134
; %bb.57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_135
.LBB1_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_136
.LBB1_59:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_137
.LBB1_60:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_138
.LBB1_61:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_139
.LBB1_62:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_64
.LBB1_63:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:8960
	global_load_b32 v37, v[35:36], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:384
.LBB1_64:                               ; %.preheader496.1.i
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v25, v26 offset1:20
	ds_store_2addr_b32 v59, v27, v28 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v29, v30 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v31, v32 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 16, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v25
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s20, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	ds_load_b32 v28, v0
	global_load_b32 v27, v[25:26], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:64
.LBB1_66:
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_68
; %bb.67:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	ds_load_b32 v28, v0 offset:1280
	global_load_b32 v27, v[25:26], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:64
.LBB1_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v25, 48, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v25
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_70
; %bb.69:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	ds_load_b32 v28, v0 offset:2560
	global_load_b32 v27, v[25:26], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:192
.LBB1_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_72
; %bb.71:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	ds_load_b32 v28, v0 offset:3840
	global_load_b32 v27, v[25:26], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:192
.LBB1_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v25, 0x50, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v25
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_74
; %bb.73:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	ds_load_b32 v28, v0 offset:5120
	global_load_b32 v27, v[25:26], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:320
.LBB1_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_76
; %bb.75:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	ds_load_b32 v28, v0 offset:6400
	global_load_b32 v27, v[25:26], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:320
.LBB1_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v25, 0x70, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s12, v25
	s_and_b32 s25, s10, vcc_lo
	s_and_saveexec_b32 s15, s25
	s_cbranch_execz .LBB1_78
; %bb.77:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s20, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	ds_load_b32 v28, v0 offset:7680
	global_load_b32 v27, v[25:26], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:448
.LBB1_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s15, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_80
; %bb.79:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s20, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	ds_load_b32 v28, v0 offset:8960
	global_load_b32 v27, v[25:26], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:448
.LBB1_80:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s15, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v17, v18 offset1:20
	ds_store_2addr_b32 v59, v19, v20 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v21, v22 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v23, v24 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_140
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_141
.LBB1_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_142
.LBB1_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_143
.LBB1_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_144
.LBB1_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_145
.LBB1_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_146
.LBB1_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_89
.LBB1_88:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:8960
	global_load_b32 v19, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:448
.LBB1_89:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v9, v10 offset1:20
	ds_store_2addr_b32 v59, v11, v12 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v13, v14 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v15, v16 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_147
; %bb.90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_148
.LBB1_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_149
.LBB1_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_150
.LBB1_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_151
.LBB1_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_152
.LBB1_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_153
.LBB1_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_98
.LBB1_97:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:8960
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
.LBB1_98:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v1, v2 offset1:20
	ds_store_2addr_b32 v59, v3, v4 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v5, v6 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v7, v8 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_154
; %bb.99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_155
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_156
.LBB1_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_157
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_158
.LBB1_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_159
.LBB1_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_160
.LBB1_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_107
.LBB1_106:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	ds_load_b32 v0, v0 offset:8960
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	global_store_b32 v[1:2], v0, off offset:448
.LBB1_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
	s_and_b32 vcc_lo, exec_lo, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_252
.LBB1_108:
	s_and_b32 vcc_lo, exec_lo, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_252
; %bb.109:                              ; %.preheader508.i18
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
	v_cmpx_gt_u32_e32 0x100, v68
	s_cbranch_execz .LBB1_113
; %bb.110:                              ; %.lr.ph.i235
	v_lshrrev_b32_e32 v0, 1, v68
	v_and_b32_e32 v1, 1, v68
	v_mov_b32_e32 v3, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v4, s22, v0
	v_lshlrev_b32_e32 v2, 2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v4
	s_cbranch_execz .LBB1_112
; %bb.111:
	v_mad_co_i64_i32 v[3:4], null, 0x48, v4, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	global_load_b32 v3, v[3:4], off
.LBB1_112:                              ; %.preheader502.loopexit.i236
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, s11, v0
	v_lshlrev_b32_e32 v1, 4, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v4, s2, v6
	v_cmp_gt_i32_e32 vcc_lo, s12, v6
	v_mad_co_i64_i32 v[4:5], null, s3, v4, s[16:17]
	global_load_b32 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v1, v1, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_f16_e32 v1, v1.l
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v1, 0, v1 :: v_dual_lshlrev_b32 v0, 3, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v0, 0, v0, v2
	ds_store_2addr_stride64_b32 v0, v3, v1 offset0:48 offset1:56
.LBB1_113:                              ; %Flow1281
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v31, 2, v68
	v_dual_mov_b32 v5, 0 :: v_dual_and_b32 v0, 3, v68
	s_add_co_i32 s4, s14, -1
	v_lshlrev_b32_e32 v17, 4, v68
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v32, s22, v31
	v_dual_mov_b32 v8, 0 :: v_dual_add_nc_u32 v1, s11, v31
	v_lshlrev_b32_e32 v0, 3, v0
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v59, 64, v32
	s_delay_alu instid0(VALU_DEP_3)
	v_add_nc_u32_e32 v2, 64, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v3, s4, v32
	v_min_i32_e32 v1, s2, v1
	v_min_i32_e32 v4, s4, v59
	v_or_b32_e32 v49, 8, v69
	v_cmp_gt_i32_e64 s0, s14, v32
	v_cmp_gt_i32_e64 s1, s14, v59
	v_mad_co_u64_u32 v[232:233], null, 0x48, v3, v[0:1]
	v_mov_b32_e32 v3, 0
	v_min_i32_e32 v2, s2, v2
	v_mad_co_u64_u32 v[182:183], null, s3, v1, v[0:1]
	v_mad_co_u64_u32 v[183:184], null, 0x48, v4, v[0:1]
	v_mov_b32_e32 v4, 0
	v_dual_mov_b32 v55, 0 :: v_dual_and_b32 v50, 16, v17
	v_mad_co_u64_u32 v[184:185], null, s3, v2, v[0:1]
	global_load_b64 v[25:26], v232, s[18:19] offset:8
	global_load_b64 v[27:28], v182, s[16:17] offset:8
	global_load_b64 v[29:30], v183, s[18:19] offset:8
	global_load_b64 v[57:58], v184, s[16:17] offset:8
	v_bfe_u32 v0, v68, 1, 1
	v_and_or_b32 v31, v31, 15, v50
	v_mov_b32_e32 v20, 0
	v_bfe_u32 v67, v68, 4, 1
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v70, 15, v68
	v_and_or_b32 v50, v69, 6, v0
	v_lshlrev_b32_e32 v31, 3, v31
	v_and_or_b32 v0, v49, 14, v0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v37, 0
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v60, v50, 8, v31
	v_lshl_or_b32 v0, v0, 8, v31
	v_mov_b32_e32 v52, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v35, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v251, 0, v60
	v_add_nc_u32_e32 v252, 0, v0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v62, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v61, 0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v26, 0, v26, s0
	v_cndmask_b32_e64 v25, 0, v25, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v60, 0, v30, s1
	v_cndmask_b32_e64 v59, 0, v29, s1
	v_mov_b32_e32 v29, 0
	ds_store_2addr_stride64_b64 v251, v[25:26], v[27:28] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v252, v[59:60], v[57:58] offset1:8
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_mov_b32_e32 v26, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_162
; %bb.114:                              ; %.preheader501.lr.ph.i191
	v_lshrrev_b32_e32 v0, 1, v68
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v2, 31, v68
	v_bfe_u32 v3, v68, 5, 1
	v_lshlrev_b32_e32 v7, 3, v70
	v_and_b32_e32 v9, 1, v68
	v_lshrrev_b32_e32 v4, 6, v68
	v_add_nc_u32_e32 v5, s11, v0
	v_lshlrev_b32_e32 v10, 3, v2
	v_lshl_or_b32 v11, v3, 9, v7
	v_add_nc_u32_e32 v7, s22, v0
	v_lshlrev_b32_e32 v0, 3, v0
	v_lshlrev_b32_e32 v12, 2, v9
	v_lshlrev_b32_e32 v2, 8, v4
	v_lshl_or_b32 v253, v4, 10, v10
	v_min_i32_e32 v4, s4, v7
	v_lshlrev_b32_e32 v6, 6, v67
	v_add3_u32 v0, 0, v0, v12
	v_mov_b32_e32 v66, v232
	s_ashr_i32 s15, s14, 31
	s_clause 0x3                            ; 16-byte Folded Spill
	scratch_store_b32 off, v4, off offset:36
	scratch_store_b32 off, v67, off offset:60
	scratch_store_b32 off, v68, off offset:52
	scratch_store_b32 off, v0, off offset:40
	v_lshl_add_u32 v0, v3, 11, 0
	v_dual_mov_b32 v4, v1 :: v_dual_lshlrev_b32 v3, 4, v9
	v_lshlrev_b32_e32 v9, 2, v9
	s_movk_i32 s28, 0x1000
	s_movk_i32 s13, 0x3000
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v3, off offset:44
	scratch_store_b32 off, v70, off offset:64
	v_mov_b32_e32 v3, v1
	v_add3_u32 v2, 0, v2, v6
	v_dual_mov_b32 v6, v1 :: v_dual_add_nc_u32 v11, 0, v11
	v_mov_b32_e32 v65, v1
	s_movk_i32 s23, 0x3800
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v2, off offset:8
	scratch_store_b64 off, v[66:67], off
	v_mov_b32_e32 v2, v1
	v_min_i32_e32 v8, s2, v5
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v65, off offset:16
	scratch_store_b32 off, v65, off offset:20
	v_mad_co_u64_u32 v[13:14], null, s3, v8, s[16:17]
	v_ashrrev_i32_e32 v8, 31, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[14:15], null, s3, v8, v[14:15]
	v_mov_b32_e32 v8, v1
	v_cmp_gt_i32_e64 s2, s14, v7
	v_cmp_gt_i32_e64 s3, s12, v5
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v40, v8
	s_clause 0x1                            ; 16-byte Folded Spill
	scratch_store_b32 off, v11, off offset:12
	scratch_store_b96 off, v[13:15], off offset:24
	v_dual_mov_b32 v35, v3 :: v_dual_add_nc_u32 v0, v0, v10
	v_mov_b32_e32 v38, v6
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v69, off offset:56
	scratch_store_b32 off, v9, off offset:48
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v13, v5
	v_dual_mov_b32 v48, v8 :: v_dual_mov_b32 v45, v5
	v_dual_mov_b32 v24, v8 :: v_dual_mov_b32 v21, v5
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v53, v5
	v_dual_mov_b32 v32, v8 :: v_dual_mov_b32 v29, v5
	v_dual_mov_b32 v64, v8 :: v_dual_mov_b32 v61, v5
	v_mov_b32_e32 v39, v7
	v_dual_mov_b32 v37, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v34, v2
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v47, v7 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v42, v2
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v2
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v50, v2
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v2
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v60, v4
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v58, v2
	s_branch .LBB1_116
.LBB1_115:                              ;   in Loop: Header=BB1_116 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s6, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_161
.LBB1_116:                              ; %.preheader501.i195
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_118 Depth 2
	s_mov_b32 s7, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s24, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s24, s10
	s_mov_b32 s26, -1
	s_cselect_b32 s25, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s27, s5
	s_mov_b32 s29, s5
	s_branch .LBB1_118
.LBB1_117:                              ;   in Loop: Header=BB1_118 Depth=2
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v178, v223, v179
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v173, v173
	v_fmac_f32_e32 v64, v178, v177
	v_mul_f32_e32 v177, v222, v179
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v165, v165
	v_fmac_f32_e32 v63, v177, v176
	v_mul_f32_e32 v176, v221, v179
	v_cvt_f32_i32_e32 v170, v170
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v166, v166
	v_fmac_f32_e32 v62, v176, v175
	v_mul_f32_e32 v175, v220, v179
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v163, v163
	v_fmac_f32_e32 v61, v175, v174
	v_mul_f32_e32 v174, v219, v179
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v159, v159
	v_fmac_f32_e32 v60, v174, v173
	v_mul_f32_e32 v173, v218, v179
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v156, v156
	v_fmac_f32_e32 v59, v173, v172
	v_mul_f32_e32 v172, v217, v179
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v150, v150
	v_fmac_f32_e32 v58, v172, v171
	v_mul_f32_e32 v171, v223, v255
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v147, v147
	v_fmac_f32_e32 v56, v171, v170
	v_mul_f32_e32 v170, v222, v255
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v143, v143
	v_fmac_f32_e32 v55, v170, v169
	v_mul_f32_e32 v169, v221, v255
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v140, v140
	v_fmac_f32_e32 v54, v169, v168
	v_mul_f32_e32 v168, v220, v255
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v135, v135
	v_fmac_f32_e32 v53, v168, v167
	v_mul_f32_e32 v167, v219, v255
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v127, v127
	v_fmac_f32_e32 v52, v167, v166
	v_mul_f32_e32 v166, v218, v255
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v118, v118
	v_fmac_f32_e32 v51, v166, v165
	v_mul_f32_e32 v165, v217, v255
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v116, v116
	v_fmac_f32_e32 v50, v165, v164
	v_mul_f32_e32 v164, v223, v248
	v_cvt_f32_i32_e32 v117, v117
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v113, v113
	v_dual_fmac_f32 v48, v164, v163 :: v_dual_mul_f32 v163, v222, v248
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v109, v109
	v_dual_fmac_f32 v47, v163, v162 :: v_dual_mul_f32 v162, v221, v248
	v_cvt_f32_i32_e32 v102, v102
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v111, v111
	v_dual_fmac_f32 v46, v162, v161 :: v_dual_mul_f32 v161, v220, v248
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v114, v114
	v_fmac_f32_e32 v45, v161, v160
	v_mul_f32_e32 v160, v219, v248
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v103, v103
	v_dual_fmac_f32 v44, v160, v159 :: v_dual_mul_f32 v159, v218, v248
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v94, v94
	v_dual_fmac_f32 v43, v159, v158 :: v_dual_mul_f32 v158, v217, v248
	v_cvt_f32_i32_e32 v88, v88
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v96, v96
	s_wait_loadcnt_dscnt 0x0
	v_fmac_f32_e32 v42, v158, v157
	v_mul_f32_e32 v157, v223, v249
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v74, v74
	v_fmac_f32_e32 v40, v157, v156
	v_mul_f32_e32 v156, v222, v249
	v_cvt_f32_i32_e32 v83, v83
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v81, v81
	v_fmac_f32_e32 v39, v156, v155
	v_mul_f32_e32 v155, v221, v249
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v93, v93
	v_fmac_f32_e32 v38, v155, v154
	v_mul_f32_e32 v154, v220, v249
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v76, v76
	v_fmac_f32_e32 v37, v154, v153
	v_mul_f32_e32 v153, v219, v249
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v78, v78
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v86, v86
	v_fmac_f32_e32 v36, v153, v152
	v_mul_f32_e32 v152, v218, v249
	v_cvt_f32_i32_e32 v73, v73
	v_cvt_f32_i32_e32 v71, v71
	v_cvt_f32_i32_e32 v69, v69
	v_cvt_f32_i32_e32 v75, v75
	v_fmac_f32_e32 v35, v152, v151
	v_mul_f32_e32 v151, v217, v249
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v66, v66
	v_cvt_f32_i32_e32 v67, v67
	v_cvt_f32_i32_e32 v79, v79
	v_fmac_f32_e32 v34, v151, v150
	v_mul_f32_e32 v150, v179, v231
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v68, v68
	v_cvt_f32_i32_e32 v70, v70
	s_xor_b32 s4, s26, -1
	v_dual_fmac_f32 v32, v150, v149 :: v_dual_mul_f32 v149, v179, v230
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s27, -1
	v_dual_fmac_f32 v31, v149, v148 :: v_dual_mul_f32 v148, v179, v227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v148, v147 :: v_dual_mul_f32 v147, v179, v226
	v_fmac_f32_e32 v29, v147, v146
	v_mul_f32_e32 v146, v179, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v146, v145 :: v_dual_mul_f32 v145, v179, v224
	v_dual_fmac_f32 v27, v145, v144 :: v_dual_mul_f32 v144, v179, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v144, v143
	v_mul_f32_e32 v143, v255, v231
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v142, v255, v230
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v142, v141
	v_mul_f32_e32 v141, v255, v227
	v_fmac_f32_e32 v22, v141, v140
	v_mul_f32_e32 v140, v255, v226
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v140, v139
	v_mul_f32_e32 v139, v255, v225
	v_fmac_f32_e32 v20, v139, v138
	v_mul_f32_e32 v138, v255, v224
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v138, v137
	v_mul_f32_e32 v137, v255, v229
	v_fmac_f32_e32 v18, v137, v136
	v_mul_f32_e32 v136, v248, v231
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v136, v135
	v_mul_f32_e32 v135, v248, v230
	v_dual_fmac_f32 v15, v135, v134 :: v_dual_mul_f32 v134, v248, v227
	v_mul_f32_e32 v135, v249, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v14, v134, v133 :: v_dual_mul_f32 v133, v248, v226
	v_mul_f32_e32 v134, v249, v224
	v_dual_fmac_f32 v13, v133, v132 :: v_dual_mul_f32 v132, v248, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v3, v134, v123
	v_mul_f32_e32 v133, v249, v227
	v_fmac_f32_e32 v12, v132, v131
	v_dual_mul_f32 v131, v248, v224 :: v_dual_mul_f32 v132, v249, v226
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v6, v133, v126
	v_dual_fmac_f32 v11, v131, v130 :: v_dual_mul_f32 v130, v248, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v131, v249, v231 :: v_dual_fmac_f32 v10, v130, v129
	v_mul_f32_e32 v129, v249, v229
	v_fmac_f32_e32 v2, v129, v122
	v_mul_f32_e32 v122, v239, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v236, v228
	v_dual_mul_f32 v130, v249, v230 :: v_dual_fmac_f32 v59, v115, v116
	v_mul_f32_e32 v115, v237, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v7, v130, v127 :: v_dual_fmac_f32 v60, v115, v117
	v_mul_f32_e32 v115, v234, v228
	v_fmac_f32_e32 v5, v132, v125
	v_fmac_f32_e32 v61, v115, v118
	v_mul_f32_e32 v115, v235, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v62, v115, v119 :: v_dual_mul_f32 v115, v232, v228
	v_fmac_f32_e32 v63, v115, v120
	v_mul_f32_e32 v115, v233, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v64, v115, v121
	v_mul_f32_e32 v115, v239, v254
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v236, v254
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v8, v131, v128 :: v_dual_fmac_f32 v51, v108, v109
	v_mul_f32_e32 v108, v237, v254
	v_fmac_f32_e32 v52, v108, v110
	v_mul_f32_e32 v108, v234, v254
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v135, v124 :: v_dual_fmac_f32 v53, v108, v111
	v_mul_f32_e32 v108, v235, v254
	v_fmac_f32_e32 v54, v108, v112
	v_mul_f32_e32 v108, v232, v254
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v55, v108, v113 :: v_dual_mul_f32 v108, v233, v254
	v_fmac_f32_e32 v56, v108, v114
	v_mul_f32_e32 v108, v239, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v236, v180
	v_fmac_f32_e32 v43, v101, v102
	v_mul_f32_e32 v101, v237, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v44, v101, v103 :: v_dual_mul_f32 v101, v234, v180
	v_fmac_f32_e32 v45, v101, v104
	v_mul_f32_e32 v101, v235, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v46, v101, v105 :: v_dual_mul_f32 v101, v232, v180
	v_fmac_f32_e32 v47, v101, v106
	v_mul_f32_e32 v101, v233, v180
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v48, v101, v107 :: v_dual_mul_f32 v101, v239, v181
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v236, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v35, v94, v95 :: v_dual_mul_f32 v94, v237, v181
	v_fmac_f32_e32 v36, v94, v96
	v_mul_f32_e32 v94, v234, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v37, v94, v97
	v_mul_f32_e32 v94, v235, v181
	v_fmac_f32_e32 v38, v94, v98
	v_mul_f32_e32 v94, v232, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v39, v94, v99 :: v_dual_mul_f32 v94, v233, v181
	scratch_load_b64 v[232:233], off, off   ; 8-byte Folded Reload
	v_fmac_f32_e32 v40, v94, v100
	v_mul_f32_e32 v94, v228, v245
	v_dual_fmac_f32 v26, v94, v87 :: v_dual_mul_f32 v87, v228, v242
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v27, v87, v88
	v_mul_f32_e32 v87, v228, v243
	v_dual_fmac_f32 v28, v87, v89 :: v_dual_mul_f32 v87, v228, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v29, v87, v90
	v_mul_f32_e32 v87, v228, v241
	v_dual_fmac_f32 v30, v87, v91 :: v_dual_mul_f32 v87, v228, v246
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v31, v87, v92
	v_mul_f32_e32 v87, v228, v247
	v_fmac_f32_e32 v32, v87, v93
	v_mul_f32_e32 v87, v254, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v87, v80
	v_mul_f32_e32 v80, v254, v242
	v_dual_fmac_f32 v19, v80, v81 :: v_dual_mul_f32 v80, v254, v243
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v81, v189, v192 :: v_dual_fmac_f32 v20, v80, v82
	v_mul_f32_e32 v80, v254, v240
	v_cvt_f32_i32_e32 v82, v190
	v_dual_fmac_f32 v21, v80, v83 :: v_dual_mul_f32 v80, v254, v241
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v80, v84
	v_mul_f32_e32 v80, v254, v246
	v_dual_fmac_f32 v23, v80, v85 :: v_dual_mul_f32 v80, v254, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v80, v86
	v_mul_f32_e32 v80, v180, v245
	v_fmac_f32_e32 v10, v80, v73
	v_dual_mul_f32 v73, v180, v242 :: v_dual_mul_f32 v80, v189, v191
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v11, v73, v74
	v_dual_mul_f32 v73, v180, v243 :: v_dual_mul_f32 v74, v181, v246
	v_dual_fmac_f32 v12, v73, v75 :: v_dual_mul_f32 v73, v180, v240
	v_mul_f32_e32 v75, v181, v245
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v13, v73, v76
	v_dual_mul_f32 v73, v180, v241 :: v_dual_mul_f32 v76, v181, v242
	v_fmac_f32_e32 v2, v75, v66
	v_cvt_f32_i32_e32 v66, v216
	v_mul_f32_e32 v75, v189, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v14, v73, v77 :: v_dual_fmac_f32 v3, v76, v67
	v_mul_f32_e32 v67, v210, v215
	v_mul_f32_e32 v73, v180, v246
	v_mul_f32_e32 v77, v181, v243
	v_mul_f32_e32 v76, v189, v198
	v_fmac_f32_e32 v3, v80, v82
	v_dual_fmac_f32 v64, v67, v66 :: v_dual_mul_f32 v67, v209, v215
	v_fmac_f32_e32 v15, v73, v78
	v_dual_mul_f32 v73, v180, v247 :: v_dual_mul_f32 v78, v181, v240
	v_fmac_f32_e32 v2, v75, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v63, v67, v66
	v_mul_f32_e32 v67, v206, v215
	v_dual_fmac_f32 v7, v74, v71 :: v_dual_mul_f32 v74, v215, v192
	v_mul_f32_e32 v71, v215, v197
	v_dual_fmac_f32 v62, v67, v66 :: v_dual_mul_f32 v67, v205, v215
	v_fmac_f32_e32 v16, v73, v79
	v_mul_f32_e32 v73, v181, v247
	v_dual_mul_f32 v79, v181, v241 :: v_dual_fmac_f32 v28, v74, v66
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v61, v67, v66
	v_dual_mul_f32 v67, v204, v215 :: v_dual_fmac_f32 v8, v73, v72
	v_mul_f32_e32 v73, v215, v193
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v6, v79, v70
	v_mul_f32_e32 v70, v215, v198
	v_fmac_f32_e32 v60, v67, v66
	v_dual_mul_f32 v67, v203, v215 :: v_dual_fmac_f32 v4, v77, v68
	v_fmac_f32_e32 v5, v78, v69
	v_dual_mul_f32 v69, v215, v191 :: v_dual_mul_f32 v68, v215, v196
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v59, v67, v66
	v_dual_mul_f32 v67, v208, v215 :: v_dual_mul_f32 v72, v215, v194
	v_dual_fmac_f32 v32, v70, v66 :: v_dual_fmac_f32 v31, v71, v66
	v_dual_fmac_f32 v29, v73, v66 :: v_dual_fmac_f32 v58, v67, v66
	v_fmac_f32_e32 v27, v69, v66
	v_dual_mul_f32 v67, v207, v215 :: v_dual_fmac_f32 v26, v68, v66
	v_mul_f32_e32 v68, v213, v196
	v_mul_f32_e32 v70, v213, v198
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v74, v213, v192 :: v_dual_fmac_f32 v57, v67, v66
	v_dual_mul_f32 v67, v215, v195 :: v_dual_fmac_f32 v30, v72, v66
	v_dual_mul_f32 v72, v213, v194 :: v_dual_mul_f32 v73, v213, v193
	v_mul_f32_e32 v78, v189, v194
	v_fmac_f32_e32 v25, v67, v66
	v_cvt_f32_i32_e32 v66, v214
	v_mul_f32_e32 v67, v210, v213
	v_mul_f32_e32 v69, v213, v191
	v_mul_f32_e32 v77, v189, v197
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v79, v189, v193 :: v_dual_fmac_f32 v24, v70, v66
	v_dual_fmac_f32 v56, v67, v66 :: v_dual_mul_f32 v67, v209, v213
	v_dual_fmac_f32 v22, v72, v66 :: v_dual_fmac_f32 v21, v73, v66
	v_fmac_f32_e32 v20, v74, v66
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v18, v68, v66 :: v_dual_fmac_f32 v55, v67, v66
	v_dual_mul_f32 v67, v206, v213 :: v_dual_mul_f32 v68, v211, v196
	v_dual_mul_f32 v72, v211, v194 :: v_dual_mul_f32 v73, v211, v193
	v_dual_fmac_f32 v19, v69, v66 :: v_dual_fmac_f32 v54, v67, v66
	v_mul_f32_e32 v67, v205, v213
	v_dual_mul_f32 v69, v211, v191 :: v_dual_mul_f32 v70, v211, v198
	v_mul_f32_e32 v74, v211, v192
	v_fmac_f32_e32 v6, v78, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v53, v67, v66
	v_mul_f32_e32 v67, v204, v213
	v_dual_fmac_f32 v8, v76, v82 :: v_dual_fmac_f32 v7, v77, v82
	v_dual_fmac_f32 v5, v79, v82 :: v_dual_fmac_f32 v4, v81, v82
	v_fmac_f32_e32 v52, v67, v66
	v_mul_f32_e32 v67, v203, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v51, v67, v66
	v_mul_f32_e32 v67, v208, v213
	v_fmac_f32_e32 v50, v67, v66
	v_mul_f32_e32 v67, v207, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v49, v67, v66
	v_mul_f32_e32 v67, v213, v195
	v_fmac_f32_e32 v17, v67, v66
	v_mul_f32_e32 v67, v210, v211
	v_mul_f32_e32 v71, v213, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v71, v66
	v_cvt_f32_i32_e32 v66, v212
	v_dual_fmac_f32 v48, v67, v66 :: v_dual_mul_f32 v67, v209, v211
	v_fmac_f32_e32 v10, v68, v66
	v_fmac_f32_e32 v16, v70, v66
	v_mul_f32_e32 v70, v206, v189
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v14, v72, v66 :: v_dual_fmac_f32 v47, v67, v66
	v_fmac_f32_e32 v12, v74, v66
	v_dual_mul_f32 v67, v206, v211 :: v_dual_mul_f32 v72, v203, v189
	v_dual_mul_f32 v71, v211, v197 :: v_dual_fmac_f32 v38, v70, v82
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v13, v73, v66 :: v_dual_fmac_f32 v46, v67, v66
	v_dual_mul_f32 v67, v205, v211 :: v_dual_mul_f32 v68, v210, v189
	v_mul_f32_e32 v73, v204, v189
	v_fmac_f32_e32 v15, v71, v66
	v_fmac_f32_e32 v11, v69, v66
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v67, v66
	v_mul_f32_e32 v67, v204, v211
	v_mul_f32_e32 v71, v205, v189
	v_mul_f32_e32 v74, v189, v195
	v_fmac_f32_e32 v40, v68, v82
	v_fmac_f32_e32 v36, v73, v82
	v_fmac_f32_e32 v44, v67, v66
	v_mul_f32_e32 v67, v203, v211
	v_fmac_f32_e32 v37, v71, v82
	v_fmac_f32_e32 v35, v72, v82
	v_fmac_f32_e32 v1, v74, v82
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v43, v67, v66
	v_mul_f32_e32 v67, v208, v211
	v_fmac_f32_e32 v42, v67, v66
	v_mul_f32_e32 v67, v207, v211
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v41, v67, v66
	v_mul_f32_e32 v67, v211, v195
	v_fmac_f32_e32 v9, v67, v66
	v_dual_mul_f32 v66, v207, v189 :: v_dual_mul_f32 v67, v208, v189
	v_mul_f32_e32 v69, v209, v189
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v33, v66, v82 :: v_dual_fmac_f32 v34, v67, v82
	v_fmac_f32_e32 v39, v69, v82
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_115
.LBB1_118:                              ;   Parent Loop BB1_116 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s29, s7
	s_and_b32 s30, s26, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[4:5], s[14:15]
	s_cselect_b32 s33, s13, 0x3400
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_cselect_b32 s36, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_mov_b32 s35, s5
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v66, s34, s30, v232
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, s31, 0, s34
	s_lshl_b32 s34, s29, 6
	v_add_co_u32 v68, s30, s30, v183
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[8:9], s[34:35]
	v_add_co_ci_u32_e64 v69, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v70, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v71, null, s35, 0, s30
	v_add_co_u32 v72, s30, s34, v184
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s30
	s_clause 0x1
	global_load_b64 v[185:186], v[70:71], off offset:40
	global_load_b64 v[187:188], v[72:73], off offset:40
	scratch_load_b32 v72, off, off offset:8 ; 4-byte Folded Reload
	s_clause 0x1
	global_load_b64 v[66:67], v[66:67], off offset:40
	global_load_b64 v[68:69], v[68:69], off offset:40
	v_add_nc_u32_e32 v76, s28, v253
	ds_load_b64 v[70:71], v76
	s_wait_dscnt 0x0
	v_dual_mov_b32 v75, v71 :: v_dual_mov_b32 v74, v70
	s_wait_loadcnt 0x2
	v_add_nc_u32_e32 v250, s36, v72
	ds_load_b64 v[72:73], v0
	ds_load_2addr_b32 v[215:216], v250 offset1:2
	ds_load_2addr_b32 v[217:218], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[219:220], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[221:222], v250 offset0:12 offset1:14
	;;#ASMSTART
	v_xor_b32 v74, v74, v65
	;;#ASMEND
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v200, 0, v67, s0
	v_cndmask_b32_e64 v199, 0, v66, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v202, 0, v69, s1
	v_cndmask_b32_e64 v201, 0, v68, s1
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[170:177], v[74:75], v[72:73], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v65, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f32_i32_e32 v66, v170
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v178, s33, v65
	ds_load_b32 v227, v178
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v215, v227
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v57, v65, v66, v57
	;;#ASMSTART
	v_xor_b32 v69, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:512
	v_dual_mov_b32 v67, v70 :: v_dual_mov_b32 v68, v71
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v226, v178 offset:128
	v_cvt_f32_i32_e32 v66, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v215, v226
	v_fma_f32 v49, v65, v66, v49
	;;#ASMSTART
	v_xor_b32 v69, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1024
	v_mov_b32_e32 v67, v70
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v225, v178 offset:256
	v_cvt_f32_i32_e32 v66, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v215, v225
	v_fma_f32 v41, v65, v66, v41
	;;#ASMSTART
	v_xor_b32 v67, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1536
	;;#ASMSTART
	v_xor_b32 v70, v70, v67
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[70:71], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v224, v178 offset:384
	v_cvt_f32_i32_e32 v66, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v215, v224
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v65, v66, v33
	;;#ASMSTART
	v_xor_b32 v71, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:512
	ds_load_b64 v[67:68], v0
	ds_load_2addr_b32 v[208:209], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[210:211], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[212:213], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[214:215], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v67, v227, v208
	v_cvt_f32_i32_e32 v68, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v67, v68, v25
	;;#ASMSTART
	v_xor_b32 v71, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v67, v226, v208
	v_cvt_f32_i32_e32 v68, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v17, v67, v68, v17
	;;#ASMSTART
	v_xor_b32 v71, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v67, v225, v208
	v_cvt_f32_i32_e32 v68, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, v67, v68, v9
	;;#ASMSTART
	v_xor_b32 v69, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v224, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v66, v121
	v_fma_f32 v1, v65, v66, v1
	;;#ASMSTART
	v_xor_b32 v71, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:256
	ds_load_b64 v[67:68], v0 offset:256
	ds_load_2addr_b32 v[207:208], v250 offset1:2
	ds_load_2addr_b32 v[205:206], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[203:204], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[197:198], v250 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v223, v178
	v_cvt_f32_i32_e32 v68, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v207, v223
	v_fmac_f32_e32 v57, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178 offset:128
	v_cvt_f32_i32_e32 v68, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v207, v181
	v_fmac_f32_e32 v49, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:256
	v_cvt_f32_i32_e32 v68, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v207, v180
	v_fmac_f32_e32 v41, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v207, v179
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[228:229], v76 offset:768
	ds_load_b64 v[65:66], v0 offset:256
	ds_load_2addr_b32 v[195:196], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[193:194], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[191:192], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[189:190], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v228 :: v_dual_mov_b32 v68, v229
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v223, v195
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:768
	v_mov_b32_e32 v67, v228
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v181, v195
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1280
	v_mov_b32_e32 v67, v228
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v195
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[230:231], v0 offset:1792
	;;#ASMSTART
	v_xor_b32 v228, v228, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[228:229], v[230:231], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v195, v179, v195
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v195, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s28, s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_stride64_b64 v251, v[199:200], v[185:186] offset1:16
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_120
; %bb.119:                              ;   in Loop: Header=BB1_118 Depth=2
	scratch_load_b32 v195, off, off offset:36 ; 4-byte Folded Reload
	s_add_co_i32 s4, s4, 1
	s_mov_b32 s37, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[4:5], s[14:15]
	s_add_co_i32 s4, s29, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_mul_u64 s[34:35], s[4:5], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_xor_b32 s29, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v185, s33, s30, v232
	s_add_nc_u64 s[34:35], s[16:17], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v186, null, s31, 0, s33
	v_add_co_u32 v187, s33, s30, v183
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v188, null, s31, 0, s33
	s_wait_loadcnt 0x0
	v_mad_co_i64_i32 v[199:200], null, 0x48, v195, s[30:31]
	scratch_load_b32 v195, off, off offset:48 ; 4-byte Folded Reload
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v228, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v229, null, s35, 0, s30
	v_add_co_u32 v230, s30, s34, v184
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v231, null, s35, 0, s30
	s_wait_loadcnt 0x0
	v_add_co_u32 v232, vcc_lo, v199, v195
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v233, null, 0, v200, vcc_lo
	scratch_load_b96 v[199:201], off, off offset:24 ; 12-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_co_u32 v195, vcc_lo, v199, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v199, null, 0, v200, vcc_lo
	s_lshl_b32 s4, s29, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v234, vcc_lo, v195, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v235, null, 0, v199, vcc_lo
	s_clause 0x1
	global_load_b64 v[199:200], v[185:186], off offset:8
	global_load_b64 v[201:202], v[187:188], off offset:8
	s_clause 0x1
	global_load_b64 v[185:186], v[228:229], off offset:8
	global_load_b64 v[187:188], v[230:231], off offset:8
	global_load_b32 v195, v[232:233], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v195, off offset:16 ; 4-byte Folded Spill
	global_load_b32 v195, v[234:235], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v195, off offset:20 ; 4-byte Folded Spill
.LBB1_120:                              ; %.preheader499.i222
                                        ;   in Loop: Header=BB1_118 Depth=2
	v_dual_mul_f32 v195, v222, v227 :: v_dual_mul_f32 v228, v220, v227
	v_dual_mul_f32 v207, v221, v227 :: v_dual_mul_f32 v230, v218, v227
	v_dual_mul_f32 v229, v219, v227 :: v_dual_mul_f32 v232, v216, v227
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v231, v217, v227
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_fma_f32 v58, v232, v171, v58
	v_fma_f32 v63, v207, v176, v63
	v_fma_f32 v60, v230, v173, v60
	v_fma_f32 v61, v229, v174, v61
	v_fma_f32 v62, v228, v175, v62
	v_fmac_f32_e32 v64, v195, v177
	v_fma_f32 v59, v231, v172, v59
	v_dual_mul_f32 v171, v222, v226 :: v_dual_mul_f32 v172, v221, v226
	v_dual_mul_f32 v173, v220, v226 :: v_dual_mul_f32 v174, v219, v226
	v_dual_mul_f32 v175, v218, v226 :: v_dual_mul_f32 v176, v217, v226
	v_mul_f32_e32 v177, v216, v226
	v_cvt_f32_i32_e32 v170, v170
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_fma_f32 v50, v177, v164, v50
	v_fma_f32 v51, v176, v165, v51
	v_fma_f32 v52, v175, v166, v52
	v_fma_f32 v53, v174, v167, v53
	v_fma_f32 v54, v173, v168, v54
	v_fma_f32 v55, v172, v169, v55
	v_dual_fmac_f32 v56, v171, v170 :: v_dual_mul_f32 v165, v221, v225
	v_mul_f32_e32 v164, v222, v225
	v_dual_mul_f32 v166, v220, v225 :: v_dual_mul_f32 v167, v219, v225
	v_dual_mul_f32 v168, v218, v225 :: v_dual_mul_f32 v169, v217, v225
	v_mul_f32_e32 v170, v216, v225
	v_cvt_f32_i32_e32 v163, v163
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v161, v161
	v_fma_f32 v42, v170, v157, v42
	v_fma_f32 v43, v169, v158, v43
	v_fma_f32 v45, v167, v160, v45
	v_fma_f32 v47, v165, v162, v47
	v_dual_fmac_f32 v48, v164, v163 :: v_dual_mul_f32 v157, v222, v224
	v_mul_f32_e32 v158, v221, v224
	v_dual_mul_f32 v160, v219, v224 :: v_dual_mul_f32 v163, v216, v224
	v_mul_f32_e32 v162, v217, v224
	v_cvt_f32_i32_e32 v156, v156
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v153, v153
	v_fma_f32 v44, v168, v159, v44
	v_fma_f32 v46, v166, v161, v46
	v_mul_f32_e32 v159, v220, v224
	v_mul_f32_e32 v161, v218, v224
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v154, v154
	v_fma_f32 v34, v163, v150, v34
	v_fma_f32 v35, v162, v151, v35
	v_fma_f32 v37, v160, v153, v37
	v_fma_f32 v39, v158, v155, v39
	v_dual_fmac_f32 v40, v157, v156 :: v_dual_mul_f32 v155, v227, v210
	v_dual_mul_f32 v150, v227, v215 :: v_dual_mul_f32 v153, v227, v212
	v_mul_f32_e32 v151, v227, v214
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v146, v146
	v_fma_f32 v36, v161, v152, v36
	v_fma_f32 v38, v159, v154, v38
	v_mul_f32_e32 v152, v227, v213
	v_mul_f32_e32 v154, v227, v211
	v_mul_f32_e32 v156, v227, v209
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v147, v147
	v_fma_f32 v27, v155, v144, v27
	v_fma_f32 v29, v153, v146, v29
	v_fma_f32 v31, v151, v148, v31
	v_mul_f32_e32 v144, v226, v214
	v_mul_f32_e32 v146, v226, v212
	v_mul_f32_e32 v148, v226, v210
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v139, v139
	v_fma_f32 v26, v156, v143, v26
	v_fma_f32 v28, v154, v145, v28
	v_fmac_f32_e32 v32, v150, v149
	v_mul_f32_e32 v143, v226, v215
	v_fma_f32 v30, v152, v147, v30
	v_mul_f32_e32 v145, v226, v213
	v_mul_f32_e32 v147, v226, v211
	v_mul_f32_e32 v149, v226, v209
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v140, v140
	v_fma_f32 v21, v146, v139, v21
	v_fma_f32 v23, v144, v141, v23
	v_mul_f32_e32 v139, v225, v212
	v_mul_f32_e32 v141, v225, v210
	v_fma_f32 v19, v148, v137, v19
	v_mul_f32_e32 v137, v225, v214
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v132, v132
	v_fma_f32 v18, v149, v136, v18
	v_fma_f32 v20, v147, v138, v20
	v_fma_f32 v22, v145, v140, v22
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v136, v225, v215
	v_mul_f32_e32 v138, v225, v213
	v_mul_f32_e32 v140, v225, v211
	v_mul_f32_e32 v142, v225, v209
	v_cvt_f32_i32_e32 v135, v135
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v133, v133
	v_fma_f32 v11, v141, v130, v11
	v_fma_f32 v13, v139, v132, v13
	v_fma_f32 v15, v137, v134, v15
	v_mul_f32_e32 v130, v224, v214
	v_mul_f32_e32 v132, v224, v212
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v116, v116
	v_mul_f32_e32 v134, v224, v210
	v_fma_f32 v12, v140, v131, v12
	v_fma_f32 v14, v138, v133, v14
	v_mul_f32_e32 v131, v224, v213
	v_mul_f32_e32 v133, v224, v211
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v126, v126
	v_fma_f32 v3, v134, v123, v3
	v_mul_f32_e32 v123, v205, v223
	v_fma_f32 v7, v130, v127, v7
	v_mul_f32_e32 v127, v197, v223
	v_fma_f32 v5, v132, v125, v5
	v_mul_f32_e32 v125, v203, v223
	v_fma_f32 v10, v142, v129, v10
	v_mul_f32_e32 v129, v224, v215
	v_cvt_f32_i32_e32 v120, v120
	v_fmac_f32_e32 v16, v136, v135
	v_mul_f32_e32 v135, v224, v209
	v_cvt_f32_i32_e32 v118, v118
	v_fma_f32 v4, v133, v124, v4
	v_fma_f32 v6, v131, v126, v6
	v_fmac_f32_e32 v8, v129, v128
	v_fma_f32 v2, v135, v122, v2
	v_dual_mul_f32 v122, v208, v223 :: v_dual_fmac_f32 v61, v125, v118
	v_mul_f32_e32 v124, v206, v223
	v_mul_f32_e32 v126, v204, v223
	v_mul_f32_e32 v128, v198, v223
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v117, v117
	v_fmac_f32_e32 v63, v127, v120
	v_mul_f32_e32 v118, v203, v181
	v_cvt_f32_i32_e32 v113, v113
	v_cvt_f32_i32_e32 v111, v111
	v_cvt_f32_i32_e32 v104, v104
	v_mul_f32_e32 v120, v197, v181
	v_fmac_f32_e32 v62, v126, v119
	v_dual_fmac_f32 v60, v124, v117 :: v_dual_fmac_f32 v59, v123, v116
	v_dual_mul_f32 v116, v205, v181 :: v_dual_mul_f32 v117, v206, v181
	v_mul_f32_e32 v119, v204, v181
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v110, v110
	v_fmac_f32_e32 v55, v120, v113
	v_dual_mul_f32 v113, v197, v180 :: v_dual_fmac_f32 v64, v128, v121
	v_fmac_f32_e32 v53, v118, v111
	v_mul_f32_e32 v121, v198, v181
	v_dual_mul_f32 v111, v203, v180 :: v_dual_fmac_f32 v58, v122, v115
	v_fmac_f32_e32 v51, v116, v109
	v_mul_f32_e32 v115, v208, v181
	v_cvt_f32_i32_e32 v106, v106
	v_fmac_f32_e32 v56, v121, v114
	v_fmac_f32_e32 v54, v119, v112
	v_fmac_f32_e32 v52, v117, v110
	v_dual_mul_f32 v109, v205, v180 :: v_dual_mul_f32 v110, v206, v180
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v208, v180
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v103, v103
	v_fmac_f32_e32 v47, v113, v106
	v_fmac_f32_e32 v45, v111, v104
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v88, v88
	v_mul_f32_e32 v106, v197, v179
	v_mul_f32_e32 v112, v204, v180
	v_cvt_f32_i32_e32 v90, v90
	v_mul_f32_e32 v104, v203, v179
	v_dual_mul_f32 v114, v198, v180 :: v_dual_fmac_f32 v43, v109, v102
	v_cvt_f32_i32_e32 v95, v95
	v_fmac_f32_e32 v39, v106, v99
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v37, v104, v97 :: v_dual_fmac_f32 v48, v114, v107
	v_dual_mul_f32 v107, v198, v179 :: v_dual_fmac_f32 v42, v108, v101
	v_mul_f32_e32 v97, v223, v191
	v_mul_f32_e32 v99, v223, v189
	v_cvt_f32_i32_e32 v92, v92
	v_mul_f32_e32 v102, v205, v179
	v_fmac_f32_e32 v46, v112, v105
	v_mul_f32_e32 v105, v204, v179
	v_fmac_f32_e32 v44, v110, v103
	v_mul_f32_e32 v101, v208, v179
	v_mul_f32_e32 v103, v206, v179
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v96, v96
	v_fmac_f32_e32 v31, v99, v92
	v_fmac_f32_e32 v35, v102, v95
	v_mul_f32_e32 v95, v223, v193
	v_fmac_f32_e32 v29, v97, v90
	v_mul_f32_e32 v90, v181, v191
	v_cvt_f32_i32_e32 v83, v83
	v_cvt_f32_i32_e32 v100, v100
	v_fmac_f32_e32 v36, v103, v96
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v96, v223, v194
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v89, v89
	v_dual_fmac_f32 v27, v95, v88 :: v_dual_mul_f32 v88, v181, v193
	v_mul_f32_e32 v92, v181, v189
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v85, v85
	v_dual_fmac_f32 v38, v105, v98 :: v_dual_fmac_f32 v21, v90, v83
	v_dual_mul_f32 v83, v180, v191 :: v_dual_mul_f32 v98, v223, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v19, v88, v81 :: v_dual_fmac_f32 v40, v107, v100
	v_mul_f32_e32 v94, v223, v196
	v_dual_mul_f32 v100, v223, v190 :: v_dual_mul_f32 v81, v180, v193
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v93, v93
	v_dual_fmac_f32 v30, v98, v91 :: v_dual_mul_f32 v91, v181, v192
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v74, v74
	v_cvt_f32_i32_e32 v76, v76
	v_dual_fmac_f32 v28, v96, v89 :: v_dual_mul_f32 v89, v181, v194
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v26, v94, v87 :: v_dual_fmac_f32 v11, v81, v74
	v_cvt_f32_i32_e32 v86, v86
	v_dual_fmac_f32 v23, v92, v85 :: v_dual_fmac_f32 v20, v89, v82
	v_mul_f32_e32 v81, v179, v189
	v_mul_f32_e32 v85, v180, v189
	v_cvt_f32_i32_e32 v78, v78
	v_dual_fmac_f32 v32, v100, v93 :: v_dual_mul_f32 v93, v181, v190
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v22, v91, v84
	v_mul_f32_e32 v84, v180, v192
	v_mul_f32_e32 v87, v181, v196
	v_cvt_f32_i32_e32 v80, v80
	v_fmac_f32_e32 v24, v93, v86
	v_mul_f32_e32 v86, v180, v190
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v75, v75
	v_fmac_f32_e32 v14, v84, v77
	v_mul_f32_e32 v77, v179, v193
	v_cvt_f32_i32_e32 v84, v67
	v_mul_f32_e32 v82, v180, v194
	v_fmac_f32_e32 v18, v87, v80
	v_cvt_f32_i32_e32 v73, v73
	v_fmac_f32_e32 v16, v86, v79
	v_mul_f32_e32 v79, v179, v191
	v_cvt_f32_i32_e32 v69, v69
	v_mul_f32_e32 v80, v180, v196
	v_dual_fmac_f32 v3, v77, v84 :: v_dual_fmac_f32 v12, v82, v75
	v_add_nc_u32_e32 v75, 0, v253
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v5, v79, v69
	v_dual_fmac_f32 v10, v80, v73 :: v_dual_fmac_f32 v13, v83, v76
	v_cvt_f32_i32_e32 v83, v66
	ds_load_b64 v[73:74], v75 offset:8192
	ds_load_b64 v[66:67], v0
	ds_load_2addr_b32 v[216:217], v250 offset1:2
	ds_load_2addr_b32 v[218:219], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[220:221], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[222:223], v250 offset0:12 offset1:14
	v_dual_fmac_f32 v15, v85, v78 :: v_dual_mul_f32 v76, v179, v196
	v_mul_f32_e32 v78, v179, v194
	v_mul_f32_e32 v80, v179, v192
	v_mul_f32_e32 v82, v179, v190
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v70, v70
	v_cvt_f32_i32_e32 v85, v68
	v_fmac_f32_e32 v7, v81, v71
	v_fmac_f32_e32 v2, v76, v83
	v_fmac_f32_e32 v8, v82, v72
	v_fmac_f32_e32 v6, v80, v70
	v_fmac_f32_e32 v4, v78, v85
	s_xor_b32 s4, s28, -1
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v74 :: v_dual_mov_b32 v68, v73
	;;#ASMSTART
	v_xor_b32 v68, v68, v65
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[170:177], v[68:69], v[66:67], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178
	v_cvt_f32_i32_e32 v66, v170
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v216, v179
	v_fmac_f32_e32 v57, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:512
	v_dual_mov_b32 v67, v73 :: v_dual_mov_b32 v68, v74
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v255, v178 offset:128
	v_cvt_f32_i32_e32 v66, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v216, v255
	v_fmac_f32_e32 v49, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1024
	v_mov_b32_e32 v67, v73
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v248, v178 offset:256
	v_cvt_f32_i32_e32 v66, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v216, v248
	v_fmac_f32_e32 v41, v65, v66
	;;#ASMSTART
	v_xor_b32 v67, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1536
	;;#ASMSTART
	v_xor_b32 v73, v73, v67
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[73:74], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v249, v178 offset:384
	v_cvt_f32_i32_e32 v66, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v216, v249
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v71, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8704
	ds_load_b64 v[67:68], v0
	ds_load_2addr_b32 v[228:229], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[224:225], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[226:227], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[230:231], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v67, v179, v228
	v_cvt_f32_i32_e32 v68, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v67, v255, v228
	v_cvt_f32_i32_e32 v68, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v67, v248, v228
	v_cvt_f32_i32_e32 v68, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v249, v228
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v66, v121
	v_fmac_f32_e32 v1, v65, v66
	;;#ASMSTART
	v_xor_b32 v71, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8448
	ds_load_b64 v[67:68], v0 offset:256
	ds_load_2addr_b32 v[238:239], v250 offset1:2
	ds_load_2addr_b32 v[236:237], v250 offset0:4 offset1:6
	ds_load_2addr_b32 v[234:235], v250 offset0:8 offset1:10
	ds_load_2addr_b32 v[232:233], v250 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v228, v178
	v_cvt_f32_i32_e32 v68, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v228
	v_fmac_f32_e32 v57, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v254, v178 offset:128
	v_cvt_f32_i32_e32 v68, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v254
	v_fmac_f32_e32 v49, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:256
	v_cvt_f32_i32_e32 v68, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v180
	v_fmac_f32_e32 v41, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v0 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v238, v181
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[189:190], v75 offset:8960
	ds_load_b64 v[65:66], v0 offset:256
	ds_load_2addr_b32 v[244:245], v250 offset0:32 offset1:34
	ds_load_2addr_b32 v[242:243], v250 offset0:36 offset1:38
	ds_load_2addr_b32 v[240:241], v250 offset0:40 offset1:42
	ds_load_2addr_b32 v[246:247], v250 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v189 :: v_dual_mov_b32 v68, v190
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v228, v244
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:768
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v254, v244
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v0 offset:1280
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v244
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[191:192], v0 offset:1792
	;;#ASMSTART
	v_xor_b32 v189, v189, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[189:190], v[191:192], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v189, v181, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v189, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b32 v[207:208], v250 offset0:1 offset1:3
	ds_load_2addr_b32 v[203:204], v250 offset0:5 offset1:7
	ds_load_2addr_b32 v[205:206], v250 offset0:9 offset1:11
	ds_load_2addr_b32 v[209:210], v250 offset0:13 offset1:15
	ds_load_2addr_b32 v[215:216], v178 offset1:1
	ds_load_2addr_b32 v[213:214], v178 offset0:32 offset1:33
	ds_load_2addr_b32 v[211:212], v178 offset0:64 offset1:65
	ds_load_2addr_b32 v[189:190], v178 offset0:96 offset1:97
	ds_load_2addr_b32 v[195:196], v250 offset0:33 offset1:35
	ds_load_2addr_b32 v[191:192], v250 offset0:37 offset1:39
	ds_load_2addr_b32 v[193:194], v250 offset0:41 offset1:43
	ds_load_2addr_b32 v[197:198], v250 offset0:45 offset1:47
	s_movk_i32 s28, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_117
; %bb.121:                              ; %.preheader500.i227
                                        ;   in Loop: Header=BB1_118 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v178, off, off offset:44
	scratch_load_b32 v238, off, off offset:20
	s_and_b32 s4, s27, exec_lo
	s_cselect_b32 s4, s13, 0x3400
	v_cndmask_b32_e64 v200, 0, v200, s0
	v_cndmask_b32_e64 v199, 0, v199, s0
	v_cndmask_b32_e64 v202, 0, v202, s1
	v_cndmask_b32_e64 v201, 0, v201, s1
	s_movk_i32 s28, 0x1000
	ds_store_2addr_stride64_b64 v251, v[199:200], v[185:186] offset1:8
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:8
	scratch_load_b32 v250, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v178, v238
	scratch_load_b32 v238, off, off offset:16 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v178, v178.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v244, s4, v250
	s_cselect_b32 s4, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v250, s4, v250
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v238, 0, v238, s2
	ds_store_b32 v244, v238
	ds_store_b32 v250, v178
	s_branch .LBB1_117
.LBB1_122:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:1280
	global_load_b32 v53, v[51:52], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_38
.LBB1_123:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:2560
	global_load_b32 v53, v[51:52], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_39
.LBB1_124:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:3840
	global_load_b32 v53, v[51:52], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_40
.LBB1_125:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:5120
	global_load_b32 v53, v[51:52], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_41
.LBB1_126:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:6400
	global_load_b32 v53, v[51:52], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_42
.LBB1_127:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:7680
	global_load_b32 v53, v[51:52], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_43
	s_branch .LBB1_44
.LBB1_128:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:1280
	global_load_b32 v45, v[43:44], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_48
.LBB1_129:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:2560
	global_load_b32 v45, v[43:44], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_49
.LBB1_130:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:3840
	global_load_b32 v45, v[43:44], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_50
.LBB1_131:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:5120
	global_load_b32 v45, v[43:44], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_51
.LBB1_132:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:6400
	global_load_b32 v45, v[43:44], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_52
.LBB1_133:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:7680
	global_load_b32 v45, v[43:44], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:384
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_53
	s_branch .LBB1_54
.LBB1_134:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:1280
	global_load_b32 v37, v[35:36], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execz .LBB1_58
.LBB1_135:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:2560
	global_load_b32 v37, v[35:36], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_59
.LBB1_136:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:3840
	global_load_b32 v37, v[35:36], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:128
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_60
.LBB1_137:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:5120
	global_load_b32 v37, v[35:36], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_61
.LBB1_138:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:6400
	global_load_b32 v37, v[35:36], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_62
.LBB1_139:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:7680
	global_load_b32 v37, v[35:36], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:384
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_63
	s_branch .LBB1_64
.LBB1_140:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_82
.LBB1_141:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:1280
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_83
.LBB1_142:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:2560
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_84
.LBB1_143:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:3840
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_85
.LBB1_144:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:5120
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_86
.LBB1_145:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:6400
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_87
.LBB1_146:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:7680
	global_load_b32 v19, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_88
	s_branch .LBB1_89
.LBB1_147:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_91
.LBB1_148:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:1280
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_92
.LBB1_149:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:2560
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_93
.LBB1_150:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:3840
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_94
.LBB1_151:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:5120
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_95
.LBB1_152:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:6400
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_96
.LBB1_153:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
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
	ds_load_b32 v12, v0 offset:7680
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_97
	s_branch .LBB1_98
.LBB1_154:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_100
.LBB1_155:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:1280
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_101
.LBB1_156:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:2560
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_102
.LBB1_157:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:3840
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_103
.LBB1_158:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:5120
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_104
.LBB1_159:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:6400
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_105
.LBB1_160:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:7680
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_106
	s_branch .LBB1_107
.LBB1_161:                              ; %.preheader497.i24.loopexit
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v68, off, off offset:52
	scratch_load_b32 v69, off, off offset:56
	scratch_load_b32 v67, off, off offset:60
	scratch_load_b32 v70, off, off offset:64
.LBB1_162:                              ; %.preheader497.i24
	s_wait_loadcnt 0x2
	v_mul_u32_u24_e32 v0, 0x500, v69
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v65, 2, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v65, 0, v0, v65
	v_lshrrev_b32_e32 v0, 4, v68
	v_mad_u32_u24 v66, 0x280, v67, v65
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v0, 15, v0
	ds_store_2addr_b32 v66, v57, v58 offset1:20
	ds_store_2addr_b32 v66, v59, v60 offset0:40 offset1:60
	ds_store_2addr_b32 v66, v61, v62 offset0:80 offset1:100
	ds_store_2addr_b32 v66, v63, v64 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v57, s11, v70
	v_or_b32_e32 v60, s22, v0
	v_mul_u32_u24_e32 v58, 0x50, v70
	v_lshlrev_b32_e32 v0, 2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v57
	v_cmp_gt_i32_e32 vcc_lo, s14, v60
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v0, 0, v58, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_164
; %bb.163:
	v_mad_co_i64_i32 v[61:62], null, s12, v60, 0
	ds_load_b32 v63, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	v_add_co_u32 v61, s0, s20, v61
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, s21, v62, s0
	v_add_co_u32 v58, s0, v61, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v62, v59, s0
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v63, off
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v61, 64, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v61
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_166
; %bb.165:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v64, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off
.LBB1_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v58, 32, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v58
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_168
; %bb.167:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v64, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off offset:128
.LBB1_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_170
; %bb.169:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v64, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off offset:128
.LBB1_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v58, 64, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v58
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_172
; %bb.171:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v64, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off offset:256
.LBB1_172:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_174
; %bb.173:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v64, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off offset:256
.LBB1_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v58, 0x60, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v58
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_176
; %bb.175:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v64, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[58:59], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v58, s1, v62, v58
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v59, null, v63, v59, s1
	s_wait_dscnt 0x0
	global_store_b32 v[58:59], v64, off offset:384
.LBB1_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v59, 0x280, v67
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_178
; %bb.177:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v64, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v58, s1, s20, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s21, v63, s1
	v_add_co_u32 v62, s1, v58, v66
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v64, off offset:384
.LBB1_178:                              ; %.preheader.1.i62
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v59, v65, v59
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v49, v50 offset1:20
	ds_store_2addr_b32 v59, v51, v52 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v53, v54 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v55, v56 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v49, 16, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v49
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_180
; %bb.179:
	v_mad_co_i64_i32 v[50:51], null, s12, v49, 0
	ds_load_b32 v54, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[52:53], 2, v[57:58]
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v50, s2, s20, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, s21, v51, s2
	v_add_co_u32 v50, s2, v50, v52
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, v51, v53, s2
	s_wait_dscnt 0x0
	global_store_b32 v[50:51], v54, off
.LBB1_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v50, 0x50, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v50
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_255
; %bb.181:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_256
.LBB1_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_257
.LBB1_183:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_258
.LBB1_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_259
.LBB1_185:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_260
.LBB1_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_188
.LBB1_187:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:384
.LBB1_188:                              ; %.preheader.2.i73
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v41, v42 offset1:20
	ds_store_2addr_b32 v59, v43, v44 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v45, v46 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v47, v48 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v41
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_190
; %bb.189:
	v_mad_co_i64_i32 v[42:43], null, s12, v41, 0
	ds_load_b32 v46, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[44:45], 2, v[57:58]
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v42, s4, s20, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s21, v43, s4
	v_add_co_u32 v42, s4, v42, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, v43, v45, s4
	s_wait_dscnt 0x0
	global_store_b32 v[42:43], v46, off
.LBB1_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v42, 0x60, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v42
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_261
; %bb.191:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_262
.LBB1_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_263
.LBB1_193:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_264
.LBB1_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_265
.LBB1_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_266
.LBB1_196:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_198
.LBB1_197:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:384
.LBB1_198:                              ; %.preheader.3.i84
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v33, v34 offset1:20
	ds_store_2addr_b32 v59, v35, v36 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v37, v38 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v39, v40 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v33, 48, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v33
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB1_200
; %bb.199:
	v_mad_co_i64_i32 v[34:35], null, s12, v33, 0
	ds_load_b32 v38, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[36:37], 2, v[57:58]
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v34, s6, s20, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, s21, v35, s6
	v_add_co_u32 v34, s6, v34, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, v35, v37, s6
	s_wait_dscnt 0x0
	global_store_b32 v[34:35], v38, off
.LBB1_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v34, 0x70, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v34
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_267
; %bb.201:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_268
.LBB1_202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_269
.LBB1_203:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_270
.LBB1_204:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_271
.LBB1_205:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_272
.LBB1_206:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_208
.LBB1_207:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:384
.LBB1_208:                              ; %.preheader496.1.i95
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v25, v26 offset1:20
	ds_store_2addr_b32 v59, v27, v28 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v29, v30 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v31, v32 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 16, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v25
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_210
; %bb.209:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:64
.LBB1_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_212
; %bb.211:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:64
.LBB1_212:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v25, 48, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v25
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_214
; %bb.213:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:192
.LBB1_214:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_216
; %bb.215:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:192
.LBB1_216:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v25, 0x50, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v25
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_218
; %bb.217:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:320
.LBB1_218:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_220
; %bb.219:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s20, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:320
.LBB1_220:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v25, 0x70, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v25
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB1_222
; %bb.221:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s20, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:448
.LBB1_222:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_224
; %bb.223:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s20, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s21, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:448
.LBB1_224:                              ; %.preheader.1.1.i108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v17, v18 offset1:20
	ds_store_2addr_b32 v59, v19, v20 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v21, v22 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v23, v24 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_273
; %bb.225:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_274
.LBB1_226:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_275
.LBB1_227:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_276
.LBB1_228:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_277
.LBB1_229:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_278
.LBB1_230:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_279
.LBB1_231:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_233
.LBB1_232:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:448
.LBB1_233:                              ; %.preheader.2.1.i117
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v9, v10 offset1:20
	ds_store_2addr_b32 v59, v11, v12 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v13, v14 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v15, v16 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_280
; %bb.234:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_281
.LBB1_235:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_282
.LBB1_236:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_283
.LBB1_237:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_284
.LBB1_238:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_285
.LBB1_239:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_286
.LBB1_240:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_242
.LBB1_241:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:448
.LBB1_242:                              ; %.preheader.3.1.i126
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v1, v2 offset1:20
	ds_store_2addr_b32 v59, v3, v4 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v5, v6 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v7, v8 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_287
; %bb.243:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_288
.LBB1_244:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_289
.LBB1_245:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_290
.LBB1_246:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_291
.LBB1_247:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_292
.LBB1_248:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_293
.LBB1_249:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_251
.LBB1_250:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v4, v0 offset:8960
	v_ashrrev_i32_e32 v58, 31, v57
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 2, v[57:58]
	v_add_co_u32 v0, vcc_lo, s20, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v4, off offset:448
.LBB1_251:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
.LBB1_252:                              ; %Flow1284
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_254
; %bb.253:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_254:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB1_255:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_182
.LBB1_256:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_183
.LBB1_257:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_184
.LBB1_258:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_185
.LBB1_259:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_186
.LBB1_260:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s20, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s21, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_187
	s_branch .LBB1_188
.LBB1_261:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_192
.LBB1_262:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_193
.LBB1_263:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_194
.LBB1_264:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_195
.LBB1_265:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_196
.LBB1_266:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s20, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s21, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_197
	s_branch .LBB1_198
.LBB1_267:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB1_202
.LBB1_268:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_203
.LBB1_269:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_204
.LBB1_270:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_205
.LBB1_271:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_206
.LBB1_272:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s20, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s21, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_207
	s_branch .LBB1_208
.LBB1_273:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_226
.LBB1_274:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_227
.LBB1_275:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_228
.LBB1_276:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_229
.LBB1_277:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_230
.LBB1_278:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_231
.LBB1_279:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s20, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s21, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_232
	s_branch .LBB1_233
.LBB1_280:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_235
.LBB1_281:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_236
.LBB1_282:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_237
.LBB1_283:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_238
.LBB1_284:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_239
.LBB1_285:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_240
.LBB1_286:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s20, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_241
	s_branch .LBB1_242
.LBB1_287:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_244
.LBB1_288:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:1280
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_245
.LBB1_289:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:2560
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_246
.LBB1_290:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:3840
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_247
.LBB1_291:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:5120
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_248
.LBB1_292:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:6400
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_249
.LBB1_293:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:7680
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s20, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s21, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_250
	s_branch .LBB1_251
.Lfunc_end1:
	.size	gemm_mq4g256v2_residual_mmq_iu4, .Lfunc_end1-gemm_mq4g256v2_residual_mmq_iu4
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 72
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-gemm_mq4g256v2_residual_mmq_iu4)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_vgpr, 256
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.numbered_sgpr, 41
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.private_seg_size, 72
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 39476
; TotalNumSgprs: 43
; NumVgprs: 256
; ScratchSize: 72
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s21, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s20, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s21, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_105
; %bb.1:                                ; %.preheader508.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v65, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v66, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s3, s12, -1
	s_ashr_i32 s22, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s2, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v3, s21, v65
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v1, 2, v66
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[2:3], null, 0x48, v3, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, v2, v1
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	global_load_b32 v2, v[2:3], off
.LBB2_4:                                ; %.preheader502.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, s20, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v3, s3, v5
	v_cmp_gt_i32_e32 vcc_lo, s12, v5
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[3:4], null, s2, v3, s[4:5]
	global_load_b32 v3, v[3:4], off
	v_lshlrev_b32_e32 v4, 4, v66
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v4, 3, v65
	v_cvt_f32_f16_e32 v3, v3.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v1, 0, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	ds_store_2addr_stride64_b32 v1, v2, v3 offset0:48 offset1:56
.LBB2_5:                                ; %Flow519
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v31, 2, v0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v40, 0 :: v_dual_lshlrev_b32 v17, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v32, s21, v31
	v_add_nc_u32_e32 v2, s20, v31
	v_dual_mov_b32 v6, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_add_nc_u32_e32 v59, 64, v32
	v_min_i32_e32 v4, s8, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v3, 64, v2
	v_min_i32_e32 v2, s3, v2
	v_dual_mov_b32 v41, 0 :: v_dual_and_b32 v50, 16, v17
	v_min_i32_e32 v5, s8, v59
	v_lshrrev_b32_e32 v69, 5, v0
	v_mad_co_u64_u32 v[231:232], null, 0x48, v4, v[1:2]
	v_mov_b32_e32 v4, 0
	v_min_i32_e32 v3, s3, v3
	v_mad_co_u64_u32 v[182:183], null, s2, v2, v[1:2]
	v_mad_co_u64_u32 v[183:184], null, 0x48, v5, v[1:2]
	v_bfe_u32 v49, v0, 1, 1
	v_and_or_b32 v31, v31, 15, v50
	v_mad_co_u64_u32 v[184:185], null, s2, v3, v[1:2]
	s_wait_kmcnt 0x0
	global_load_b64 v[25:26], v231, s[6:7] offset:8
	global_load_b64 v[27:28], v182, s[4:5] offset:8
	global_load_b64 v[29:30], v183, s[6:7] offset:8
	global_load_b64 v[57:58], v184, s[4:5] offset:8
	v_or_b32_e32 v50, 8, v69
	v_and_or_b32 v60, v69, 6, v49
	v_lshlrev_b32_e32 v31, 3, v31
	v_cmp_gt_i32_e64 s0, s14, v32
	v_mov_b32_e32 v32, 0
	v_and_or_b32 v49, v50, 14, v49
	v_cmp_gt_i32_e64 s1, s14, v59
	v_lshl_or_b32 v60, v60, 8, v31
	v_bfe_u32 v68, v0, 4, 1
	v_dual_mov_b32 v38, 0 :: v_dual_and_b32 v67, 15, v0
	v_lshl_or_b32 v31, v49, 8, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v251, 0, v60
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v5, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v36, 0
	v_add_nc_u32_e32 v252, 0, v31
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_mov_b32_e32 v49, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v61, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	v_mov_b32_e32 v62, 0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v26, 0, v26, s0
	v_cndmask_b32_e64 v25, 0, v25, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v60, 0, v30, s1
	v_cndmask_b32_e64 v59, 0, v29, s1
	ds_store_2addr_stride64_b64 v251, v[25:26], v[27:28] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v252, v[59:60], v[57:58] offset1:8
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_mov_b32_e32 v60, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_mov_b32_e32 v57, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_15
; %bb.6:                                ; %.preheader501.lr.ph.i
	v_and_b32_e32 v1, 31, v0
	v_add_nc_u32_e32 v2, s20, v65
	v_bfe_u32 v3, v0, 5, 1
	v_lshlrev_b32_e32 v6, 3, v67
	v_lshrrev_b32_e32 v4, 6, v0
	v_lshlrev_b32_e32 v9, 3, v1
	v_min_i32_e32 v1, s3, v2
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	v_lshl_or_b32 v10, v3, 9, v6
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s28, 0x1000
	v_mad_co_u64_u32 v[12:13], null, s2, v1, s[4:5]
	v_ashrrev_i32_e32 v6, 31, v1
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v7, 8, v4
	v_cmp_gt_i32_e64 s3, s12, v2
	v_lshlrev_b32_e32 v8, 3, v65
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v11, 2, v66
	v_lshl_or_b32 v253, v4, 10, v9
	v_add_nc_u32_e32 v4, s21, v65
	v_lshlrev_b32_e32 v5, 6, v68
	v_mad_co_u64_u32 v[13:14], null, s2, v6, v[13:14]
	v_mov_b32_e32 v6, v1
	s_movk_i32 s13, 0x3000
	v_min_i32_e32 v0, s8, v4
	v_cmp_gt_i32_e64 s2, s14, v4
	v_mov_b32_e32 v4, v1
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	scratch_store_b32 off, v0, off offset:36 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v8, v11
	v_lshl_add_u32 v11, v3, 11, 0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v8, v1
	v_mov_b32_e32 v65, v1
	scratch_store_b32 off, v0, off offset:40 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 4, v66
	scratch_store_b32 off, v0, off offset:44 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v7, v5
	scratch_store_b32 off, v67, off offset:56 ; 4-byte Folded Spill
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v40, v8
	scratch_store_b32 off, v0, off offset:8 ; 4-byte Folded Spill
	v_dual_mov_b32 v37, v5 :: v_dual_add_nc_u32 v0, 0, v10
	scratch_store_b32 off, v69, off offset:60 ; 4-byte Folded Spill
	v_mov_b32_e32 v35, v3
	v_dual_mov_b32 v33, v1 :: v_dual_add_nc_u32 v254, v11, v9
	scratch_store_b32 off, v0, off offset:12 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 2, v66
	v_mov_b32_e32 v48, v8
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v32, v8
	scratch_store_b32 off, v0, off offset:48 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v64, v8
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:16
	scratch_store_b32 off, v0, off offset:20
	v_mov_b32_e32 v0, v231
	scratch_store_b96 off, v[12:14], off offset:24 ; 12-byte Folded Spill
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v34, v2
	v_mov_b32_e32 v9, v1
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v42, v2
	v_mov_b32_e32 v17, v1
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v18, v2
	v_mov_b32_e32 v49, v1
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v50, v2
	v_mov_b32_e32 v25, v1
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v26, v2
	v_mov_b32_e32 v57, v1
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_mov_b32_e32 v60, v4
	v_mov_b32_e32 v58, v2
	scratch_store_b64 off, v[0:1], off      ; 8-byte Folded Spill
	s_branch .LBB2_8
.LBB2_7:                                ;   in Loop: Header=BB2_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_8:                                ; %.preheader501.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s26, -1
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_mov_b32 s27, s9
	s_mov_b32 s29, s9
	s_branch .LBB2_10
.LBB2_9:                                ;   in Loop: Header=BB2_10 Depth=2
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v178, v223, v179
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v173, v173
	v_fmac_f32_e32 v64, v178, v177
	v_mul_f32_e32 v177, v222, v179
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v170, v170
	v_fmac_f32_e32 v63, v177, v176
	v_mul_f32_e32 v176, v221, v179
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_fmac_f32_e32 v62, v176, v175
	v_mul_f32_e32 v175, v220, v179
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v163, v163
	v_cvt_f32_i32_e32 v160, v160
	v_fmac_f32_e32 v61, v175, v174
	v_mul_f32_e32 v174, v219, v179
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v155, v155
	v_fmac_f32_e32 v60, v174, v173
	v_mul_f32_e32 v173, v218, v179
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v156, v156
	v_cvt_f32_i32_e32 v151, v151
	v_fmac_f32_e32 v59, v173, v172
	v_mul_f32_e32 v172, v217, v179
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v150, v150
	v_dual_fmac_f32 v58, v172, v171 :: v_dual_mul_f32 v171, v223, v180
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v147, v147
	v_fmac_f32_e32 v56, v171, v170
	v_mul_f32_e32 v170, v222, v180
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v143, v143
	v_fmac_f32_e32 v55, v170, v169
	v_mul_f32_e32 v169, v221, v180
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v140, v140
	v_cvt_f32_i32_e32 v134, v134
	v_fmac_f32_e32 v54, v169, v168
	v_mul_f32_e32 v168, v220, v180
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v130, v130
	v_fmac_f32_e32 v53, v168, v167
	v_mul_f32_e32 v167, v219, v180
	v_cvt_f32_i32_e32 v135, v135
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v131, v131
	v_fmac_f32_e32 v52, v167, v166
	v_mul_f32_e32 v166, v218, v180
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v127, v127
	v_fmac_f32_e32 v51, v166, v165
	v_mul_f32_e32 v165, v217, v180
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v117, v117
	v_fmac_f32_e32 v50, v165, v164
	v_mul_f32_e32 v164, v223, v250
	v_cvt_f32_i32_e32 v118, v118
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v113, v113
	v_cvt_f32_i32_e32 v121, v121
	v_dual_fmac_f32 v48, v164, v163 :: v_dual_mul_f32 v163, v222, v250
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v47, v163, v162
	v_mul_f32_e32 v162, v221, v250
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v111, v111
	v_cvt_f32_i32_e32 v112, v112
	v_dual_fmac_f32 v46, v162, v161 :: v_dual_mul_f32 v161, v220, v250
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v104, v104
	v_dual_fmac_f32 v45, v161, v160 :: v_dual_mul_f32 v160, v219, v250
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v103, v103
	v_cvt_f32_i32_e32 v105, v105
	v_dual_fmac_f32 v44, v160, v159 :: v_dual_mul_f32 v159, v218, v250
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v88, v88
	v_fmac_f32_e32 v43, v159, v158
	v_mul_f32_e32 v158, v217, v250
	v_cvt_f32_i32_e32 v90, v90
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v96, v96
	v_dual_fmac_f32 v42, v158, v157 :: v_dual_mul_f32 v157, v223, v248
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v83, v83
	v_fmac_f32_e32 v40, v157, v156
	v_mul_f32_e32 v156, v222, v248
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v74, v74
	v_fmac_f32_e32 v39, v156, v155
	v_mul_f32_e32 v155, v221, v248
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v85, v85
	v_fmac_f32_e32 v38, v155, v154
	v_mul_f32_e32 v154, v220, v248
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v76, v76
	v_cvt_f32_i32_e32 v78, v78
	v_fmac_f32_e32 v37, v154, v153
	v_mul_f32_e32 v153, v219, v248
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v69, v69
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v36, v153, v152
	v_mul_f32_e32 v152, v218, v248
	v_cvt_f32_i32_e32 v67, v67
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v86, v86
	v_cvt_f32_i32_e32 v73, v73
	v_fmac_f32_e32 v35, v152, v151
	v_mul_f32_e32 v151, v217, v248
	v_cvt_f32_i32_e32 v75, v75
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v70, v70
	v_fmac_f32_e32 v34, v151, v150
	v_mul_f32_e32 v150, v179, v231
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v66, v66
	v_cvt_f32_i32_e32 v68, v68
	s_xor_b32 s8, s26, -1
	v_dual_fmac_f32 v32, v150, v149 :: v_dual_mul_f32 v149, v179, v230
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s27, -1
	v_dual_fmac_f32 v31, v149, v148 :: v_dual_mul_f32 v148, v179, v227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v148, v147 :: v_dual_mul_f32 v147, v179, v226
	v_fmac_f32_e32 v29, v147, v146
	v_mul_f32_e32 v146, v179, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v146, v145 :: v_dual_mul_f32 v145, v179, v224
	v_dual_fmac_f32 v27, v145, v144 :: v_dual_mul_f32 v144, v179, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v144, v143
	v_mul_f32_e32 v143, v180, v231
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v142, v180, v230
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v142, v141
	v_mul_f32_e32 v141, v180, v227
	v_fmac_f32_e32 v22, v141, v140
	v_mul_f32_e32 v140, v180, v226
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v140, v139
	v_mul_f32_e32 v139, v180, v225
	v_fmac_f32_e32 v20, v139, v138
	v_mul_f32_e32 v138, v180, v224
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v138, v137
	v_mul_f32_e32 v137, v180, v229
	v_fmac_f32_e32 v18, v137, v136
	v_mul_f32_e32 v136, v250, v231
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v16, v136, v135 :: v_dual_mul_f32 v135, v250, v230
	v_dual_fmac_f32 v15, v135, v134 :: v_dual_mul_f32 v134, v250, v227
	v_mul_f32_e32 v135, v248, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v134, v133
	v_mul_f32_e32 v133, v250, v226
	v_dual_fmac_f32 v13, v133, v132 :: v_dual_mul_f32 v132, v250, v225
	v_mul_f32_e32 v133, v248, v227
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v12, v132, v131 :: v_dual_mul_f32 v131, v250, v224
	v_mul_f32_e32 v132, v248, v226
	v_fmac_f32_e32 v6, v133, v126
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v11, v131, v130 :: v_dual_mul_f32 v130, v250, v229
	v_dual_mul_f32 v131, v248, v231 :: v_dual_fmac_f32 v10, v130, v129
	v_mul_f32_e32 v129, v248, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v129, v122
	v_mul_f32_e32 v122, v239, v228
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v236, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v59, v115, v116
	v_mul_f32_e32 v115, v237, v228
	v_fmac_f32_e32 v5, v132, v125
	v_dual_fmac_f32 v60, v115, v117 :: v_dual_mul_f32 v115, v234, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v134, v248, v224 :: v_dual_fmac_f32 v61, v115, v118
	v_dual_mul_f32 v115, v235, v228 :: v_dual_mul_f32 v130, v248, v230
	v_fmac_f32_e32 v3, v134, v123
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v62, v115, v119 :: v_dual_mul_f32 v115, v232, v228
	v_fmac_f32_e32 v7, v130, v127
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v63, v115, v120
	v_mul_f32_e32 v115, v233, v228
	v_fmac_f32_e32 v64, v115, v121
	v_mul_f32_e32 v115, v239, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v236, v0
	v_dual_fmac_f32 v8, v131, v128 :: v_dual_fmac_f32 v51, v108, v109
	v_mul_f32_e32 v108, v237, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v108, v110
	v_mul_f32_e32 v108, v234, v0
	v_dual_fmac_f32 v4, v135, v124 :: v_dual_fmac_f32 v53, v108, v111
	v_mul_f32_e32 v108, v235, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v54, v108, v112
	v_mul_f32_e32 v108, v232, v0
	v_dual_fmac_f32 v55, v108, v113 :: v_dual_mul_f32 v108, v233, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v56, v108, v114
	v_mul_f32_e32 v108, v239, v255
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v236, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v43, v101, v102
	v_mul_f32_e32 v101, v237, v255
	v_fmac_f32_e32 v44, v101, v103
	v_mul_f32_e32 v101, v234, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v45, v101, v104
	v_mul_f32_e32 v101, v235, v255
	v_dual_fmac_f32 v46, v101, v105 :: v_dual_mul_f32 v101, v232, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v47, v101, v106
	v_mul_f32_e32 v101, v233, v255
	v_dual_fmac_f32 v48, v101, v107 :: v_dual_mul_f32 v101, v239, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v236, v181
	v_dual_fmac_f32 v35, v94, v95 :: v_dual_mul_f32 v94, v237, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v94, v96
	v_mul_f32_e32 v94, v234, v181
	v_fmac_f32_e32 v37, v94, v97
	v_mul_f32_e32 v94, v235, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v94, v98
	v_mul_f32_e32 v94, v232, v181
	scratch_load_b64 v[231:232], off, off   ; 8-byte Folded Reload
	v_dual_fmac_f32 v39, v94, v99 :: v_dual_mul_f32 v94, v233, v181
	v_fmac_f32_e32 v40, v94, v100
	v_mul_f32_e32 v94, v228, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v26, v94, v87 :: v_dual_mul_f32 v87, v228, v242
	v_fmac_f32_e32 v27, v87, v88
	v_mul_f32_e32 v87, v228, v243
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v87, v89 :: v_dual_mul_f32 v87, v228, v240
	v_fmac_f32_e32 v29, v87, v90
	v_mul_f32_e32 v87, v228, v241
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v87, v91 :: v_dual_mul_f32 v87, v228, v246
	v_fmac_f32_e32 v31, v87, v92
	v_mul_f32_e32 v87, v228, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v32, v87, v93
	v_mul_f32_e32 v87, v0, v245
	v_fmac_f32_e32 v18, v87, v80
	v_mul_f32_e32 v80, v0, v242
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v19, v80, v81
	v_mul_f32_e32 v80, v0, v243
	v_cvt_f32_i32_e32 v81, v190
	v_fmac_f32_e32 v20, v80, v82
	v_mul_f32_e32 v80, v0, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v80, v83
	v_mul_f32_e32 v80, v0, v241
	v_fmac_f32_e32 v22, v80, v84
	v_mul_f32_e32 v80, v0, v246
	v_mul_f32_e32 v0, v0, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v0, v86
	v_mul_f32_e32 v0, v255, v245
	v_fmac_f32_e32 v10, v0, v73
	v_mul_f32_e32 v0, v255, v242
	v_mul_f32_e32 v73, v181, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v11, v0, v74 :: v_dual_mul_f32 v0, v255, v243
	v_fmac_f32_e32 v7, v73, v71
	v_mul_f32_e32 v71, v215, v194
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v73, v215, v192 :: v_dual_fmac_f32 v12, v0, v75
	v_dual_mul_f32 v0, v255, v240 :: v_dual_mul_f32 v75, v181, v242
	v_dual_fmac_f32 v13, v0, v76 :: v_dual_mul_f32 v0, v255, v241
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v3, v75, v67
	v_mul_f32_e32 v75, v189, v200
	v_dual_mul_f32 v67, v215, v198 :: v_dual_fmac_f32 v14, v0, v77
	v_dual_mul_f32 v0, v255, v246 :: v_dual_mul_f32 v77, v181, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v15, v0, v78 :: v_dual_mul_f32 v0, v255, v247
	v_mul_f32_e32 v78, v181, v241
	v_fmac_f32_e32 v5, v77, v69
	v_mul_f32_e32 v77, v189, v194
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v69, v215, v200 :: v_dual_fmac_f32 v6, v78, v70
	v_mul_f32_e32 v70, v215, v199
	v_fmac_f32_e32 v16, v0, v79
	v_mul_f32_e32 v0, v181, v247
	v_dual_mul_f32 v78, v189, v193 :: v_dual_mul_f32 v79, v189, v191
	v_fmac_f32_e32 v8, v0, v72
	v_cvt_f32_i32_e32 v0, v216
	v_mul_f32_e32 v72, v215, v193
	v_mul_f32_e32 v74, v181, v245
	v_dual_fmac_f32 v5, v78, v81 :: v_dual_mul_f32 v76, v181, v243
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v31, v70, v0
	v_dual_fmac_f32 v29, v72, v0 :: v_dual_mul_f32 v72, v213, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v2, v74, v66
	v_mul_f32_e32 v66, v210, v215
	v_mul_f32_e32 v70, v213, v199
	v_fmac_f32_e32 v4, v76, v68
	v_mul_f32_e32 v68, v215, v191
	v_fmac_f32_e32 v32, v69, v0
	v_fmac_f32_e32 v64, v66, v0
	v_mul_f32_e32 v66, v209, v215
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v73, v0 :: v_dual_fmac_f32 v27, v68, v0
	v_fmac_f32_e32 v26, v67, v0
	v_dual_mul_f32 v76, v189, v199 :: v_dual_fmac_f32 v63, v66, v0
	v_dual_mul_f32 v66, v206, v215 :: v_dual_mul_f32 v67, v213, v198
	v_dual_mul_f32 v68, v213, v191 :: v_dual_mul_f32 v69, v213, v200
	v_mul_f32_e32 v73, v213, v192
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v62, v66, v0
	v_mul_f32_e32 v66, v205, v215
	v_dual_mul_f32 v74, v189, v198 :: v_dual_fmac_f32 v7, v76, v81
	v_fmac_f32_e32 v3, v79, v81
	v_dual_fmac_f32 v61, v66, v0 :: v_dual_mul_f32 v66, v204, v215
	v_dual_fmac_f32 v23, v80, v85 :: v_dual_fmac_f32 v30, v71, v0
	v_mul_f32_e32 v71, v213, v194
	v_mul_f32_e32 v80, v189, v192
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v60, v66, v0
	v_mul_f32_e32 v66, v203, v215
	v_dual_fmac_f32 v59, v66, v0 :: v_dual_mul_f32 v66, v208, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v58, v66, v0
	v_mul_f32_e32 v66, v207, v215
	v_dual_fmac_f32 v57, v66, v0 :: v_dual_mul_f32 v66, v215, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v66, v0
	v_cvt_f32_i32_e32 v0, v214
	v_mul_f32_e32 v66, v210, v213
	v_fmac_f32_e32 v24, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v66, v0
	v_dual_mul_f32 v66, v209, v213 :: v_dual_fmac_f32 v23, v70, v0
	v_dual_fmac_f32 v22, v71, v0 :: v_dual_fmac_f32 v21, v72, v0
	v_dual_fmac_f32 v20, v73, v0 :: v_dual_fmac_f32 v55, v66, v0
	v_dual_mul_f32 v66, v206, v213 :: v_dual_fmac_f32 v19, v68, v0
	v_fmac_f32_e32 v18, v67, v0
	v_dual_mul_f32 v67, v211, v198 :: v_dual_mul_f32 v68, v211, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v54, v66, v0
	v_dual_mul_f32 v66, v205, v213 :: v_dual_mul_f32 v69, v211, v200
	v_dual_mul_f32 v70, v211, v199 :: v_dual_mul_f32 v71, v211, v194
	v_dual_mul_f32 v72, v211, v193 :: v_dual_fmac_f32 v53, v66, v0
	v_dual_mul_f32 v66, v204, v213 :: v_dual_mul_f32 v73, v211, v192
	v_fmac_f32_e32 v6, v77, v81
	v_fmac_f32_e32 v4, v80, v81
	v_fmac_f32_e32 v2, v74, v81
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v66, v0
	v_mul_f32_e32 v66, v203, v213
	v_dual_fmac_f32 v51, v66, v0 :: v_dual_mul_f32 v66, v208, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v50, v66, v0
	v_mul_f32_e32 v66, v207, v213
	v_dual_fmac_f32 v49, v66, v0 :: v_dual_mul_f32 v66, v213, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v66, v0
	v_cvt_f32_i32_e32 v0, v212
	v_mul_f32_e32 v66, v210, v211
	v_fmac_f32_e32 v16, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v48, v66, v0
	v_dual_mul_f32 v66, v209, v211 :: v_dual_fmac_f32 v15, v70, v0
	v_dual_fmac_f32 v14, v71, v0 :: v_dual_fmac_f32 v13, v72, v0
	v_dual_fmac_f32 v12, v73, v0 :: v_dual_fmac_f32 v47, v66, v0
	v_dual_mul_f32 v66, v206, v211 :: v_dual_fmac_f32 v11, v68, v0
	v_dual_fmac_f32 v10, v67, v0 :: v_dual_mul_f32 v67, v210, v189
	v_mul_f32_e32 v68, v209, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v46, v66, v0
	v_dual_mul_f32 v66, v205, v211 :: v_dual_mul_f32 v69, v206, v189
	v_dual_mul_f32 v70, v205, v189 :: v_dual_mul_f32 v71, v203, v189
	v_dual_mul_f32 v72, v204, v189 :: v_dual_fmac_f32 v45, v66, v0
	v_dual_mul_f32 v66, v204, v211 :: v_dual_mul_f32 v73, v189, v197
	v_dual_fmac_f32 v40, v67, v81 :: v_dual_fmac_f32 v39, v68, v81
	v_fmac_f32_e32 v38, v69, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v66, v0
	v_dual_mul_f32 v66, v203, v211 :: v_dual_fmac_f32 v37, v70, v81
	v_dual_fmac_f32 v36, v72, v81 :: v_dual_fmac_f32 v35, v71, v81
	v_dual_fmac_f32 v8, v75, v81 :: v_dual_fmac_f32 v43, v66, v0
	v_dual_mul_f32 v66, v208, v211 :: v_dual_fmac_f32 v1, v73, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v42, v66, v0
	v_mul_f32_e32 v66, v207, v211
	v_dual_fmac_f32 v41, v66, v0 :: v_dual_mul_f32 v66, v211, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v9, v66, v0 :: v_dual_mul_f32 v0, v207, v189
	v_mul_f32_e32 v66, v208, v189
	v_dual_fmac_f32 v33, v0, v81 :: v_dual_fmac_f32 v34, v66, v81
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_7
.LBB2_10:                               ;   Parent Loop BB2_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	scratch_load_b32 v0, off, off offset:8  ; 4-byte Folded Reload
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s29, s11
	s_and_b32 s30, s26, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[8:9], s[14:15]
	s_cselect_b32 s33, s13, 0x3400
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_cselect_b32 s36, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_mov_b32 s35, s9
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v66, s34, s30, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, s31, 0, s34
	s_lshl_b32 s34, s29, 6
	v_add_co_u32 v68, s30, s30, v183
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	v_add_co_ci_u32_e64 v69, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v70, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v71, null, s35, 0, s30
	v_add_co_u32 v72, s30, s34, v184
	s_clause 0x1
	global_load_b64 v[66:67], v[66:67], off offset:40
	global_load_b64 v[68:69], v[68:69], off offset:40
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s30
	s_clause 0x1
	global_load_b64 v[185:186], v[70:71], off offset:40
	global_load_b64 v[187:188], v[72:73], off offset:40
	v_add_nc_u32_e32 v76, s28, v253
	ds_load_b64 v[70:71], v76
	s_wait_dscnt 0x0
	v_dual_mov_b32 v75, v71 :: v_dual_mov_b32 v74, v70
	s_wait_loadcnt 0x4
	v_add_nc_u32_e32 v249, s36, v0
	ds_load_b64 v[72:73], v254
	ds_load_2addr_b32 v[215:216], v249 offset1:2
	ds_load_2addr_b32 v[217:218], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[219:220], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[221:222], v249 offset0:12 offset1:14
	;;#ASMSTART
	v_xor_b32 v74, v74, v65
	;;#ASMEND
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[170:177], v[74:75], v[72:73], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v196, 0, v67, s0
	v_cndmask_b32_e64 v195, 0, v66, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v202, 0, v69, s1
	v_cndmask_b32_e64 v201, 0, v68, s1
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v0, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f32_i32_e32 v65, v170
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v178, s33, v0
	ds_load_b32 v226, v178
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v226
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v57, v0, v65, v57
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v70 :: v_dual_mov_b32 v68, v71
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v225, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v225
	v_fma_f32 v49, v0, v65, v49
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v70
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v224, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v224
	v_fma_f32 v41, v0, v65, v41
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v70, v70, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[70:71], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v223, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v223
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v0, v65, v33
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:512
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[208:209], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[210:211], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[212:213], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[214:215], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v226, v208
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v0, v67, v25
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v225, v208
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v17, v0, v67, v17
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v224, v208
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, v0, v67, v9
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v223, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fma_f32 v1, v0, v65, v1
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:256
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[207:208], v249 offset1:2
	ds_load_2addr_b32 v[205:206], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[203:204], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[199:200], v249 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v181
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:128
	v_cvt_f32_i32_e32 v67, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v180
	v_fmac_f32_e32 v49, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178 offset:256
	v_cvt_f32_i32_e32 v67, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v179
	v_fmac_f32_e32 v41, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v207, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[227:228], v76 offset:768
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[197:198], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[193:194], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[191:192], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[189:190], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v227 :: v_dual_mov_b32 v68, v228
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v181, v197
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v197
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v179, v197
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[229:230], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v227, v227, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[227:228], v[229:230], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v197, v0, v197
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v197, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s28, s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:16
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v195, off, off offset:36
	scratch_load_b32 v197, off, off offset:48
	s_add_co_i32 s8, s8, 1
	s_mov_b32 s37, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[8:9], s[14:15]
	s_add_co_i32 s8, s29, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_mul_u64 s[34:35], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_xor_b32 s29, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v185, s33, s30, v231
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v186, null, s31, 0, s33
	v_add_co_u32 v187, s33, s30, v183
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v188, null, s31, 0, s33
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[195:196], null, 0x48, v195, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v227, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v228, null, s35, 0, s30
	v_add_co_u32 v229, s30, s34, v184
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v230, null, s35, 0, s30
	s_wait_loadcnt 0x0
	v_add_co_u32 v231, vcc_lo, v195, v197
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v232, null, 0, v196, vcc_lo
	scratch_load_b96 v[195:197], off, off offset:24 ; 12-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_co_u32 v195, vcc_lo, v195, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v196, null, 0, v196, vcc_lo
	s_lshl_b32 s8, s29, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v233, vcc_lo, v195, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v234, null, 0, v196, vcc_lo
	s_clause 0x1
	global_load_b64 v[195:196], v[185:186], off offset:8
	global_load_b64 v[201:202], v[187:188], off offset:8
	s_clause 0x1
	global_load_b64 v[185:186], v[227:228], off offset:8
	global_load_b64 v[187:188], v[229:230], off offset:8
	global_load_b32 v197, v[231:232], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:16 ; 4-byte Folded Spill
	global_load_b32 v197, v[233:234], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:20 ; 4-byte Folded Spill
.LBB2_12:                               ; %.preheader499.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	v_dual_mul_f32 v197, v222, v226 :: v_dual_mul_f32 v228, v219, v226
	v_mul_f32_e32 v207, v221, v226
	v_dual_mul_f32 v227, v220, v226 :: v_dual_mul_f32 v230, v217, v226
	v_mul_f32_e32 v231, v216, v226
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v229, v218, v226
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_fmac_f32_e32 v64, v197, v177
	v_fma_f32 v59, v230, v172, v59
	v_fma_f32 v60, v229, v173, v60
	v_fma_f32 v61, v228, v174, v61
	v_fma_f32 v62, v227, v175, v62
	v_mul_f32_e32 v172, v221, v225
	v_fma_f32 v63, v207, v176, v63
	v_mul_f32_e32 v174, v219, v225
	v_fma_f32 v58, v231, v171, v58
	v_dual_mul_f32 v171, v222, v225 :: v_dual_mul_f32 v176, v217, v225
	v_mul_f32_e32 v175, v218, v225
	v_mul_f32_e32 v177, v216, v225
	v_cvt_f32_i32_e32 v170, v170
	v_mul_f32_e32 v173, v220, v225
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_fma_f32 v50, v177, v164, v50
	v_fma_f32 v51, v176, v165, v51
	v_fma_f32 v52, v175, v166, v52
	v_fma_f32 v53, v174, v167, v53
	v_fma_f32 v55, v172, v169, v55
	v_fmac_f32_e32 v56, v171, v170
	v_fma_f32 v54, v173, v168, v54
	v_dual_mul_f32 v164, v222, v224 :: v_dual_mul_f32 v165, v221, v224
	v_mul_f32_e32 v168, v218, v224
	v_dual_mul_f32 v166, v220, v224 :: v_dual_mul_f32 v167, v219, v224
	v_mul_f32_e32 v170, v216, v224
	v_cvt_f32_i32_e32 v163, v163
	v_mul_f32_e32 v169, v217, v224
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v162, v162
	v_fma_f32 v42, v170, v157, v42
	v_fma_f32 v44, v168, v159, v44
	v_fma_f32 v45, v167, v160, v45
	v_fma_f32 v46, v166, v161, v46
	v_fmac_f32_e32 v48, v164, v163
	v_fma_f32 v43, v169, v158, v43
	v_dual_mul_f32 v157, v222, v223 :: v_dual_mul_f32 v158, v221, v223
	v_dual_mul_f32 v159, v220, v223 :: v_dual_mul_f32 v160, v219, v223
	v_mul_f32_e32 v161, v218, v223
	v_cvt_f32_i32_e32 v156, v156
	v_mul_f32_e32 v163, v216, v223
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v153, v153
	v_fma_f32 v47, v165, v162, v47
	v_mul_f32_e32 v162, v217, v223
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v154, v154
	v_fma_f32 v36, v161, v152, v36
	v_fma_f32 v37, v160, v153, v37
	v_fmac_f32_e32 v40, v157, v156
	v_fma_f32 v34, v163, v150, v34
	v_mul_f32_e32 v150, v226, v215
	v_dual_mul_f32 v152, v226, v213 :: v_dual_mul_f32 v153, v226, v212
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v146, v146
	v_fma_f32 v35, v162, v151, v35
	v_fma_f32 v38, v159, v154, v38
	v_fma_f32 v39, v158, v155, v39
	v_dual_mul_f32 v151, v226, v214 :: v_dual_mul_f32 v154, v226, v211
	v_mul_f32_e32 v155, v226, v210
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v147, v147
	v_fma_f32 v29, v153, v146, v29
	v_fmac_f32_e32 v32, v150, v149
	v_dual_mul_f32 v146, v225, v212 :: v_dual_mul_f32 v149, v225, v209
	v_cvt_f32_i32_e32 v136, v136
	v_mul_f32_e32 v156, v226, v209
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v145, v145
	v_fma_f32 v27, v155, v144, v27
	v_fma_f32 v30, v152, v147, v30
	v_fma_f32 v31, v151, v148, v31
	v_dual_mul_f32 v144, v225, v214 :: v_dual_mul_f32 v147, v225, v211
	v_mul_f32_e32 v148, v225, v210
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_fma_f32 v18, v149, v136, v18
	v_mul_f32_e32 v136, v224, v215
	v_cvt_f32_i32_e32 v135, v135
	v_fma_f32 v26, v156, v143, v26
	v_fma_f32 v28, v154, v145, v28
	v_mul_f32_e32 v143, v225, v215
	v_mul_f32_e32 v145, v225, v213
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v140, v140
	v_fma_f32 v20, v147, v138, v20
	v_fma_f32 v21, v146, v139, v21
	v_fma_f32 v23, v144, v141, v23
	v_dual_mul_f32 v138, v224, v213 :: v_dual_mul_f32 v141, v224, v210
	v_mul_f32_e32 v139, v224, v212
	v_fma_f32 v19, v148, v137, v19
	v_mul_f32_e32 v137, v224, v214
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v130, v130
	v_dual_fmac_f32 v16, v136, v135 :: v_dual_mul_f32 v135, v223, v209
	v_cvt_f32_i32_e32 v122, v122
	v_fma_f32 v22, v145, v140, v22
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v140, v224, v211
	v_mul_f32_e32 v142, v224, v209
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_fma_f32 v11, v141, v130, v11
	v_fma_f32 v15, v137, v134, v15
	v_mul_f32_e32 v130, v223, v214
	v_mul_f32_e32 v134, v223, v210
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v123, v123
	v_fma_f32 v2, v135, v122, v2
	v_mul_f32_e32 v122, v208, v181
	v_cvt_f32_i32_e32 v115, v115
	v_fma_f32 v10, v142, v129, v10
	v_fma_f32 v12, v140, v131, v12
	v_fma_f32 v13, v139, v132, v13
	v_fma_f32 v14, v138, v133, v14
	v_mul_f32_e32 v129, v223, v215
	v_mul_f32_e32 v131, v223, v213
	v_mul_f32_e32 v133, v223, v211
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v126, v126
	v_fma_f32 v7, v130, v127, v7
	v_mul_f32_e32 v127, v199, v181
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v118, v118
	v_mul_f32_e32 v132, v223, v212
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v208, v180
	v_fma_f32 v3, v134, v123, v3
	v_mul_f32_e32 v123, v205, v181
	v_fma_f32 v4, v133, v124, v4
	v_fma_f32 v5, v132, v125, v5
	v_fma_f32 v6, v131, v126, v6
	v_dual_fmac_f32 v8, v129, v128 :: v_dual_mul_f32 v125, v203, v181
	v_dual_mul_f32 v124, v206, v181 :: v_dual_fmac_f32 v63, v127, v120
	v_dual_mul_f32 v126, v204, v181 :: v_dual_fmac_f32 v59, v123, v116
	v_mul_f32_e32 v128, v200, v181
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v117, v117
	v_mul_f32_e32 v120, v199, v180
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v113, v113
	v_mul_f32_e32 v116, v205, v180
	v_dual_fmac_f32 v62, v126, v119 :: v_dual_mul_f32 v119, v204, v180
	v_dual_fmac_f32 v61, v125, v118 :: v_dual_fmac_f32 v60, v124, v117
	v_mul_f32_e32 v117, v206, v180
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v111, v111
	v_dual_fmac_f32 v55, v120, v113 :: v_dual_mul_f32 v118, v203, v180
	v_fmac_f32_e32 v51, v116, v109
	v_fmac_f32_e32 v64, v128, v121
	v_mul_f32_e32 v121, v200, v180
	v_mul_f32_e32 v113, v199, v179
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v110, v110
	v_fmac_f32_e32 v56, v121, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v119, v112 :: v_dual_fmac_f32 v47, v113, v106
	v_fmac_f32_e32 v53, v118, v111
	v_dual_mul_f32 v111, v203, v179 :: v_dual_mul_f32 v112, v204, v179
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v104, v104
	v_mul_f32_e32 v106, v199, v0
	v_cvt_f32_i32_e32 v99, v99
	v_fmac_f32_e32 v52, v117, v110
	v_dual_mul_f32 v109, v205, v179 :: v_dual_mul_f32 v110, v206, v179
	v_mul_f32_e32 v114, v200, v179
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v208, v179
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v103, v103
	v_dual_fmac_f32 v46, v112, v105 :: v_dual_fmac_f32 v39, v106, v99
	v_fmac_f32_e32 v45, v111, v104
	v_dual_mul_f32 v104, v203, v0 :: v_dual_mul_f32 v105, v204, v0
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v97, v97
	v_mul_f32_e32 v99, v181, v189
	v_cvt_f32_i32_e32 v92, v92
	v_fmac_f32_e32 v48, v114, v107
	v_dual_fmac_f32 v44, v110, v103 :: v_dual_fmac_f32 v43, v109, v102
	v_dual_mul_f32 v102, v205, v0 :: v_dual_mul_f32 v103, v206, v0
	v_mul_f32_e32 v107, v200, v0
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v95, v95
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v208, v0
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v96, v96
	v_dual_fmac_f32 v38, v105, v98 :: v_dual_fmac_f32 v31, v99, v92
	v_fmac_f32_e32 v37, v104, v97
	v_mul_f32_e32 v97, v181, v191
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v74, v74
	v_mul_f32_e32 v92, v180, v189
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v107, v100 :: v_dual_fmac_f32 v29, v97, v90
	v_fmac_f32_e32 v36, v103, v96
	v_dual_mul_f32 v96, v181, v194 :: v_dual_fmac_f32 v23, v92, v85
	v_mul_f32_e32 v98, v181, v192
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v88, v88
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v181, v198
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v91, v91
	v_dual_mul_f32 v90, v180, v191 :: v_dual_mul_f32 v85, v179, v189
	v_cvt_f32_i32_e32 v83, v83
	v_mul_f32_e32 v100, v181, v190
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v21, v90, v83 :: v_dual_fmac_f32 v32, v100, v93
	v_mul_f32_e32 v83, v179, v191
	v_dual_mul_f32 v93, v180, v190 :: v_dual_fmac_f32 v30, v98, v91
	v_dual_mul_f32 v91, v180, v192 :: v_dual_fmac_f32 v26, v94, v87
	v_mul_f32_e32 v87, v180, v198
	v_fmac_f32_e32 v35, v102, v95
	v_mul_f32_e32 v95, v181, v193
	v_cvt_f32_i32_e32 v78, v78
	v_fmac_f32_e32 v28, v96, v89
	v_mul_f32_e32 v89, v180, v194
	v_cvt_f32_i32_e32 v76, v76
	v_dual_fmac_f32 v27, v95, v88 :: v_dual_mul_f32 v88, v180, v193
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v86, v86
	v_fmac_f32_e32 v20, v89, v82
	v_mul_f32_e32 v82, v179, v194
	v_cvt_f32_i32_e32 v75, v75
	v_fmac_f32_e32 v13, v83, v76
	v_fmac_f32_e32 v19, v88, v81
	v_dual_mul_f32 v81, v179, v193 :: v_dual_fmac_f32 v24, v93, v86
	v_fmac_f32_e32 v18, v87, v80
	v_cvt_f32_i32_e32 v73, v73
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v12, v82, v75 :: v_dual_fmac_f32 v11, v81, v74
	v_add_nc_u32_e32 v75, 0, v253
	v_mul_f32_e32 v81, v0, v189
	v_cvt_f32_i32_e32 v71, v71
	v_mul_f32_e32 v80, v179, v198
	v_cvt_f32_i32_e32 v82, v66
	v_cvt_f32_i32_e32 v83, v67
	ds_load_b64 v[66:67], v254
	v_dual_fmac_f32 v7, v81, v71 :: v_dual_fmac_f32 v10, v80, v73
	ds_load_b64 v[73:74], v75 offset:8192
	ds_load_2addr_b32 v[216:217], v249 offset1:2
	ds_load_2addr_b32 v[218:219], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[220:221], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[222:223], v249 offset0:12 offset1:14
	v_cvt_f32_i32_e32 v84, v84
	v_mul_f32_e32 v86, v179, v190
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v69, v69
	v_fmac_f32_e32 v22, v91, v84
	v_mul_f32_e32 v84, v179, v192
	v_fmac_f32_e32 v16, v86, v79
	v_mul_f32_e32 v79, v0, v191
	v_fmac_f32_e32 v15, v85, v78
	v_mul_f32_e32 v76, v0, v198
	v_fmac_f32_e32 v14, v84, v77
	v_dual_mul_f32 v77, v0, v193 :: v_dual_mul_f32 v78, v0, v194
	v_mul_f32_e32 v80, v0, v192
	v_mul_f32_e32 v0, v0, v190
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v70, v70
	v_cvt_f32_i32_e32 v84, v68
	v_fmac_f32_e32 v5, v79, v69
	s_wait_dscnt 0x4
	v_dual_mov_b32 v69, v74 :: v_dual_mov_b32 v68, v73
	v_fmac_f32_e32 v8, v0, v72
	v_fmac_f32_e32 v6, v80, v70
	v_dual_fmac_f32 v4, v78, v84 :: v_dual_fmac_f32 v3, v77, v83
	v_fmac_f32_e32 v2, v76, v82
	;;#ASMSTART
	v_xor_b32 v68, v68, v65
	;;#ASMEND
	v_wmma_i32_16x16x32_iu4 v[170:177], v[68:69], v[66:67], 0 neg_lo:[0,1,0]
	s_xor_b32 s8, s28, -1
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178
	v_cvt_f32_i32_e32 v65, v170
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v179
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v57, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v73 :: v_dual_mov_b32 v68, v74
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v180
	v_fmac_f32_e32 v49, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v73
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v250, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v250
	v_fmac_f32_e32 v41, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v73, v73, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[73:74], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v248, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v248
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8704
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[228:229], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[224:225], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[226:227], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[230:231], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v179, v228
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v180, v228
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v250, v228
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v248, v228
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fmac_f32_e32 v1, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8448
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[238:239], v249 offset1:2
	ds_load_2addr_b32 v[236:237], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[234:235], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[232:233], v249 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v228, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v238, v228
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:128
	v_cvt_f32_i32_e32 v68, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v0
	v_fmac_f32_e32 v49, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v255, v178 offset:256
	v_cvt_f32_i32_e32 v68, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v255
	v_fmac_f32_e32 v41, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v238, v181
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[189:190], v75 offset:8960
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[244:245], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[242:243], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[240:241], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[246:247], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v189 :: v_dual_mov_b32 v68, v190
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v228, v244
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v0, v244
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v255, v244
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[191:192], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v189, v189, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[189:190], v[191:192], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v189, v181, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v189, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b32 v[207:208], v249 offset0:1 offset1:3
	ds_load_2addr_b32 v[203:204], v249 offset0:5 offset1:7
	ds_load_2addr_b32 v[205:206], v249 offset0:9 offset1:11
	ds_load_2addr_b32 v[209:210], v249 offset0:13 offset1:15
	ds_load_2addr_b32 v[215:216], v178 offset1:1
	ds_load_2addr_b32 v[213:214], v178 offset0:32 offset1:33
	ds_load_2addr_b32 v[211:212], v178 offset0:64 offset1:65
	ds_load_2addr_b32 v[189:190], v178 offset0:96 offset1:97
	ds_load_2addr_b32 v[197:198], v249 offset0:33 offset1:35
	ds_load_2addr_b32 v[191:192], v249 offset0:37 offset1:39
	ds_load_2addr_b32 v[193:194], v249 offset0:41 offset1:43
	ds_load_2addr_b32 v[199:200], v249 offset0:45 offset1:47
	s_movk_i32 s28, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_9
; %bb.13:                               ; %.preheader500.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v178, off, off offset:44
	scratch_load_b32 v238, off, off offset:20
	s_and_b32 s8, s27, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v196, 0, v196, s0
	v_cndmask_b32_e64 v195, 0, v195, s0
	v_cndmask_b32_e64 v202, 0, v202, s1
	v_cndmask_b32_e64 v201, 0, v201, s1
	s_movk_i32 s28, 0x1000
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:8
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:8
	scratch_load_b32 v249, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v178, v238
	scratch_load_b32 v238, off, off offset:16 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v178, v178.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v244, s8, v249
	s_cselect_b32 s8, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v249, s8, v249
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v238, 0, v238, s2
	ds_store_b32 v244, v238
	ds_store_b32 v249, v178
	s_branch .LBB2_9
.LBB2_14:                               ; %.preheader497.i.loopexit
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v0, off, off offset:52
	scratch_load_b32 v67, off, off offset:56
	scratch_load_b32 v69, off, off offset:60
	s_wait_loadcnt 0x2
	v_bfe_u32 v68, v0, 4, 1
.LBB2_15:                               ; %.preheader497.i
	s_wait_loadcnt 0x0
	v_mul_u32_u24_e32 v66, 0x500, v69
	v_lshlrev_b32_e32 v65, 2, v67
	v_lshrrev_b32_e32 v0, 4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v65, 0, v66, v65
	v_and_b32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_u32_u24 v66, 0x280, v68, v65
	ds_store_2addr_b32 v66, v57, v58 offset1:20
	ds_store_2addr_b32 v66, v59, v60 offset0:40 offset1:60
	ds_store_2addr_b32 v66, v61, v62 offset0:80 offset1:100
	ds_store_2addr_b32 v66, v63, v64 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v57, s20, v67
	v_or_b32_e32 v60, s21, v0
	v_mul_u32_u24_e32 v59, 0x50, v67
	v_lshlrev_b32_e32 v0, 2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v57
	v_cmp_gt_i32_e32 vcc_lo, s14, v60
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v0, 0, v59, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_17
; %bb.16:
	v_mad_co_i64_i32 v[61:62], null, s12, v60, 0
	v_lshlrev_b64_e32 v[63:64], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	v_add_co_u32 v59, s0, s16, v61
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, s17, v62, s0
	v_add_co_u32 v61, s0, v59, v63
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v62, null, v62, v64, s0
	ds_load_b32 v63, v0
	global_load_b32 v59, v[61:62], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v63, v59
	global_store_b32 v[61:62], v59, off
.LBB2_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v61, 64, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v61
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_19
; %bb.18:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:1280
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off
.LBB2_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 32, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v59
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_21
; %bb.20:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:2560
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:128
.LBB2_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_23
; %bb.22:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:3840
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:128
.LBB2_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 64, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v59
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_25
; %bb.24:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:5120
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:256
.LBB2_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_27
; %bb.26:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:6400
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:256
.LBB2_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 0x60, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v59
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_29
; %bb.28:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	ds_load_b32 v64, v0 offset:7680
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v59, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v59, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	global_load_b32 v59, v[62:63], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v59, v64, v59
	global_store_b32 v[62:63], v59, off offset:384
.LBB2_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v59, 0x280, v68
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_31
; %bb.30:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	ds_load_b32 v66, v0 offset:8960
	global_load_b32 v64, v[62:63], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v64, v66, v64
	global_store_b32 v[62:63], v64, off offset:384
.LBB2_31:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v59, v65, v59
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v49, v50 offset1:20
	ds_store_2addr_b32 v59, v51, v52 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v53, v54 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v55, v56 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v49, 16, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v49
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_33
; %bb.32:
	v_mad_co_i64_i32 v[50:51], null, s12, v49, 0
	v_lshlrev_b64_e32 v[52:53], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	v_add_co_u32 v50, s2, s16, v50
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, s17, v51, s2
	v_add_co_u32 v50, s2, v50, v52
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v51, null, v51, v53, s2
	ds_load_b32 v53, v0
	global_load_b32 v52, v[50:51], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v52, v53, v52
	global_store_b32 v[50:51], v52, off
.LBB2_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v50, 0x50, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v50
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_106
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_107
.LBB2_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_108
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_109
.LBB2_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_110
.LBB2_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_111
.LBB2_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_41
.LBB2_40:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:8960
	global_load_b32 v53, v[51:52], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:384
.LBB2_41:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v41, v42 offset1:20
	ds_store_2addr_b32 v59, v43, v44 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v45, v46 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v47, v48 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v41
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_43
; %bb.42:
	v_mad_co_i64_i32 v[42:43], null, s12, v41, 0
	v_lshlrev_b64_e32 v[44:45], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	v_add_co_u32 v42, s4, s16, v42
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, s17, v43, s4
	v_add_co_u32 v42, s4, v42, v44
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v43, null, v43, v45, s4
	ds_load_b32 v45, v0
	global_load_b32 v44, v[42:43], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v44, v45, v44
	global_store_b32 v[42:43], v44, off
.LBB2_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v42, 0x60, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v42
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_112
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_113
.LBB2_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_114
.LBB2_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_115
.LBB2_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_116
.LBB2_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_117
.LBB2_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_51
.LBB2_50:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:8960
	global_load_b32 v45, v[43:44], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:384
.LBB2_51:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v33, v34 offset1:20
	ds_store_2addr_b32 v59, v35, v36 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v37, v38 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v39, v40 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v33, 48, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v33
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB2_53
; %bb.52:
	v_mad_co_i64_i32 v[34:35], null, s12, v33, 0
	v_lshlrev_b64_e32 v[36:37], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	v_add_co_u32 v34, s6, s16, v34
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, s17, v35, s6
	v_add_co_u32 v34, s6, v34, v36
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v35, null, v35, v37, s6
	ds_load_b32 v37, v0
	global_load_b32 v36, v[34:35], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v36, v37, v36
	global_store_b32 v[34:35], v36, off
.LBB2_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v34, 0x70, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v34
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_118
; %bb.54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_119
.LBB2_55:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_120
.LBB2_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_121
.LBB2_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_122
.LBB2_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_123
.LBB2_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_61
.LBB2_60:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:8960
	global_load_b32 v37, v[35:36], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:384
.LBB2_61:                               ; %.preheader496.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v25, v26 offset1:20
	ds_store_2addr_b32 v59, v27, v28 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v29, v30 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v31, v32 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 16, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v25
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_63
; %bb.62:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	ds_load_b32 v28, v0
	global_load_b32 v27, v[25:26], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:64
.LBB2_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_65
; %bb.64:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s8, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s8
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	ds_load_b32 v28, v0 offset:1280
	global_load_b32 v27, v[25:26], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:64
.LBB2_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v25, 48, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v25
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_67
; %bb.66:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	ds_load_b32 v28, v0 offset:2560
	global_load_b32 v27, v[25:26], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:192
.LBB2_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_69
; %bb.68:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s9, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s9
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	ds_load_b32 v28, v0 offset:3840
	global_load_b32 v27, v[25:26], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:192
.LBB2_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v25, 0x50, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v25
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_71
; %bb.70:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s16, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	ds_load_b32 v28, v0 offset:5120
	global_load_b32 v27, v[25:26], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:320
.LBB2_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_73
; %bb.72:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, s10, s16, v25
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s10
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	ds_load_b32 v28, v0 offset:6400
	global_load_b32 v27, v[25:26], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:320
.LBB2_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v25, 0x70, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v25
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB2_75
; %bb.74:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s16, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	ds_load_b32 v28, v0 offset:7680
	global_load_b32 v27, v[25:26], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:448
.LBB2_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_77
; %bb.76:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	v_add_co_u32 v25, vcc_lo, s16, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, s17, v26, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	ds_load_b32 v28, v0 offset:8960
	global_load_b32 v27, v[25:26], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v27, v28, v27
	global_store_b32 v[25:26], v27, off offset:448
.LBB2_77:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v17, v18 offset1:20
	ds_store_2addr_b32 v59, v19, v20 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v21, v22 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v23, v24 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_124
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_125
.LBB2_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_126
.LBB2_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_127
.LBB2_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_128
.LBB2_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_129
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_130
.LBB2_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_86
.LBB2_85:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:8960
	global_load_b32 v19, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:448
.LBB2_86:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v9, v10 offset1:20
	ds_store_2addr_b32 v59, v11, v12 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v13, v14 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v15, v16 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_131
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_132
.LBB2_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_133
.LBB2_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_134
.LBB2_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_135
.LBB2_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_136
.LBB2_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_137
.LBB2_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_95
.LBB2_94:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:8960
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
.LBB2_95:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v1, v2 offset1:20
	ds_store_2addr_b32 v59, v3, v4 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v5, v6 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v7, v8 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_138
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_139
.LBB2_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_140
.LBB2_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_141
.LBB2_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_142
.LBB2_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_143
.LBB2_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_144
.LBB2_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_104
.LBB2_103:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	ds_load_b32 v0, v0 offset:8960
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	global_store_b32 v[1:2], v0, off offset:448
.LBB2_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_105:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB2_106:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:1280
	global_load_b32 v53, v[51:52], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_35
.LBB2_107:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:2560
	global_load_b32 v53, v[51:52], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_36
.LBB2_108:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:3840
	global_load_b32 v53, v[51:52], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_37
.LBB2_109:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:5120
	global_load_b32 v53, v[51:52], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_38
.LBB2_110:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:6400
	global_load_b32 v53, v[51:52], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_39
.LBB2_111:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	ds_load_b32 v54, v0 offset:7680
	global_load_b32 v53, v[51:52], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v53, v54, v53
	global_store_b32 v[51:52], v53, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_40
	s_branch .LBB2_41
.LBB2_112:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:1280
	global_load_b32 v45, v[43:44], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_45
.LBB2_113:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:2560
	global_load_b32 v45, v[43:44], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_46
.LBB2_114:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:3840
	global_load_b32 v45, v[43:44], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_47
.LBB2_115:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:5120
	global_load_b32 v45, v[43:44], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_48
.LBB2_116:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:6400
	global_load_b32 v45, v[43:44], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_49
.LBB2_117:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	ds_load_b32 v46, v0 offset:7680
	global_load_b32 v45, v[43:44], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v45, v46, v45
	global_store_b32 v[43:44], v45, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_50
	s_branch .LBB2_51
.LBB2_118:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:1280
	global_load_b32 v37, v[35:36], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB2_55
.LBB2_119:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:2560
	global_load_b32 v37, v[35:36], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_56
.LBB2_120:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:3840
	global_load_b32 v37, v[35:36], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_57
.LBB2_121:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:5120
	global_load_b32 v37, v[35:36], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_58
.LBB2_122:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:6400
	global_load_b32 v37, v[35:36], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_59
.LBB2_123:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	ds_load_b32 v38, v0 offset:7680
	global_load_b32 v37, v[35:36], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v37, v38, v37
	global_store_b32 v[35:36], v37, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_60
	s_branch .LBB2_61
.LBB2_124:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_79
.LBB2_125:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:1280
	global_load_b32 v19, v[17:18], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_80
.LBB2_126:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:2560
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_81
.LBB2_127:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:3840
	global_load_b32 v19, v[17:18], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_82
.LBB2_128:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:5120
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_83
.LBB2_129:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:6400
	global_load_b32 v19, v[17:18], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_84
.LBB2_130:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	ds_load_b32 v20, v0 offset:7680
	global_load_b32 v19, v[17:18], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v20, v19
	global_store_b32 v[17:18], v19, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_85
	s_branch .LBB2_86
.LBB2_131:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_88
.LBB2_132:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:1280
	global_load_b32 v11, v[9:10], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_89
.LBB2_133:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:2560
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_90
.LBB2_134:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:3840
	global_load_b32 v11, v[9:10], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_91
.LBB2_135:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:5120
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_92
.LBB2_136:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:6400
	global_load_b32 v11, v[9:10], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_93
.LBB2_137:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v12, v0 offset:7680
	global_load_b32 v11, v[9:10], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_94
	s_branch .LBB2_95
.LBB2_138:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_97
.LBB2_139:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:1280
	global_load_b32 v3, v[1:2], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_98
.LBB2_140:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:2560
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_99
.LBB2_141:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:3840
	global_load_b32 v3, v[1:2], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_100
.LBB2_142:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:5120
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_101
.LBB2_143:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:6400
	global_load_b32 v3, v[1:2], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_102
.LBB2_144:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	ds_load_b32 v4, v0 offset:7680
	global_load_b32 v3, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v4, v3
	global_store_b32 v[1:2], v3, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_103
	s_branch .LBB2_104
.Lfunc_end2:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_add, .Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4_full_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 68
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4_full_add)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_vgpr, 256
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.private_seg_size, 68
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20072
; TotalNumSgprs: 40
; NumVgprs: 256
; ScratchSize: 68
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s21, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s20, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s21, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_105
; %bb.1:                                ; %.preheader508.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v65, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v66, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s3, s12, -1
	s_ashr_i32 s22, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s2, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v3, s21, v65
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v1, 2, v66
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v3
	s_cbranch_execz .LBB3_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[2:3], null, 0x48, v3, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v2, vcc_lo, v2, v1
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	global_load_b32 v2, v[2:3], off
.LBB3_4:                                ; %.preheader502.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, s20, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v3, s3, v5
	v_cmp_gt_i32_e32 vcc_lo, s12, v5
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[3:4], null, s2, v3, s[4:5]
	global_load_b32 v3, v[3:4], off
	v_lshlrev_b32_e32 v4, 4, v66
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshrrev_b32_e32 v3, v4, v3
	v_lshlrev_b32_e32 v4, 3, v65
	v_cvt_f32_f16_e32 v3, v3.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v1, 0, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	ds_store_2addr_stride64_b32 v1, v2, v3 offset0:48 offset1:56
.LBB3_5:                                ; %Flow519
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v31, 2, v0
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v40, 0 :: v_dual_lshlrev_b32 v17, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v32, s21, v31
	v_add_nc_u32_e32 v2, s20, v31
	v_dual_mov_b32 v6, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_add_nc_u32_e32 v59, 64, v32
	v_min_i32_e32 v4, s8, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v3, 64, v2
	v_min_i32_e32 v2, s3, v2
	v_dual_mov_b32 v41, 0 :: v_dual_and_b32 v50, 16, v17
	v_min_i32_e32 v5, s8, v59
	v_lshrrev_b32_e32 v69, 5, v0
	v_mad_co_u64_u32 v[231:232], null, 0x48, v4, v[1:2]
	v_mov_b32_e32 v4, 0
	v_min_i32_e32 v3, s3, v3
	v_mad_co_u64_u32 v[182:183], null, s2, v2, v[1:2]
	v_mad_co_u64_u32 v[183:184], null, 0x48, v5, v[1:2]
	v_bfe_u32 v49, v0, 1, 1
	v_and_or_b32 v31, v31, 15, v50
	v_mad_co_u64_u32 v[184:185], null, s2, v3, v[1:2]
	s_wait_kmcnt 0x0
	global_load_b64 v[25:26], v231, s[6:7] offset:8
	global_load_b64 v[27:28], v182, s[4:5] offset:8
	global_load_b64 v[29:30], v183, s[6:7] offset:8
	global_load_b64 v[57:58], v184, s[4:5] offset:8
	v_or_b32_e32 v50, 8, v69
	v_and_or_b32 v60, v69, 6, v49
	v_lshlrev_b32_e32 v31, 3, v31
	v_cmp_gt_i32_e64 s0, s14, v32
	v_mov_b32_e32 v32, 0
	v_and_or_b32 v49, v50, 14, v49
	v_cmp_gt_i32_e64 s1, s14, v59
	v_lshl_or_b32 v60, v60, 8, v31
	v_bfe_u32 v68, v0, 4, 1
	v_dual_mov_b32 v38, 0 :: v_dual_and_b32 v67, 15, v0
	v_lshl_or_b32 v31, v49, 8, v31
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v251, 0, v60
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v5, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v36, 0
	v_add_nc_u32_e32 v252, 0, v31
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v14, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v12, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v10, 0 :: v_dual_mov_b32 v15, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v13, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v11, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_mov_b32_e32 v49, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v61, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	v_mov_b32_e32 v62, 0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v26, 0, v26, s0
	v_cndmask_b32_e64 v25, 0, v25, s0
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v60, 0, v30, s1
	v_cndmask_b32_e64 v59, 0, v29, s1
	ds_store_2addr_stride64_b64 v251, v[25:26], v[27:28] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v252, v[59:60], v[57:58] offset1:8
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_mov_b32_e32 v60, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_mov_b32_e32 v57, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_15
; %bb.6:                                ; %.preheader501.lr.ph.i
	v_and_b32_e32 v1, 31, v0
	v_add_nc_u32_e32 v2, s20, v65
	v_bfe_u32 v3, v0, 5, 1
	v_lshlrev_b32_e32 v6, 3, v67
	v_lshrrev_b32_e32 v4, 6, v0
	v_lshlrev_b32_e32 v9, 3, v1
	v_min_i32_e32 v1, s3, v2
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	v_lshl_or_b32 v10, v3, 9, v6
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s28, 0x1000
	v_mad_co_u64_u32 v[12:13], null, s2, v1, s[4:5]
	v_ashrrev_i32_e32 v6, 31, v1
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v7, 8, v4
	v_cmp_gt_i32_e64 s3, s12, v2
	v_lshlrev_b32_e32 v8, 3, v65
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v2, v1 :: v_dual_lshlrev_b32 v11, 2, v66
	v_lshl_or_b32 v253, v4, 10, v9
	v_add_nc_u32_e32 v4, s21, v65
	v_lshlrev_b32_e32 v5, 6, v68
	v_mad_co_u64_u32 v[13:14], null, s2, v6, v[13:14]
	v_mov_b32_e32 v6, v1
	s_movk_i32 s13, 0x3000
	v_min_i32_e32 v0, s8, v4
	v_cmp_gt_i32_e64 s2, s14, v4
	v_mov_b32_e32 v4, v1
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	scratch_store_b32 off, v0, off offset:36 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v8, v11
	v_lshl_add_u32 v11, v3, 11, 0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v8, v1
	v_mov_b32_e32 v65, v1
	scratch_store_b32 off, v0, off offset:40 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 4, v66
	scratch_store_b32 off, v0, off offset:44 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v7, v5
	scratch_store_b32 off, v67, off offset:56 ; 4-byte Folded Spill
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v40, v8
	scratch_store_b32 off, v0, off offset:8 ; 4-byte Folded Spill
	v_dual_mov_b32 v37, v5 :: v_dual_add_nc_u32 v0, 0, v10
	scratch_store_b32 off, v69, off offset:60 ; 4-byte Folded Spill
	v_mov_b32_e32 v35, v3
	v_dual_mov_b32 v33, v1 :: v_dual_add_nc_u32 v254, v11, v9
	scratch_store_b32 off, v0, off offset:12 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 2, v66
	v_mov_b32_e32 v48, v8
	v_mov_b32_e32 v24, v8
	v_mov_b32_e32 v56, v8
	v_mov_b32_e32 v32, v8
	scratch_store_b32 off, v0, off offset:48 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v64, v8
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v0, off offset:16
	scratch_store_b32 off, v0, off offset:20
	v_mov_b32_e32 v0, v231
	scratch_store_b96 off, v[12:14], off offset:24 ; 12-byte Folded Spill
	v_dual_mov_b32 v16, v8 :: v_dual_mov_b32 v47, v7
	v_dual_mov_b32 v39, v7 :: v_dual_mov_b32 v38, v6
	v_dual_mov_b32 v13, v5 :: v_dual_mov_b32 v36, v4
	v_dual_mov_b32 v11, v3 :: v_dual_mov_b32 v34, v2
	v_mov_b32_e32 v9, v1
	v_dual_mov_b32 v15, v7 :: v_dual_mov_b32 v14, v6
	v_dual_mov_b32 v45, v5 :: v_dual_mov_b32 v12, v4
	v_dual_mov_b32 v43, v3 :: v_dual_mov_b32 v10, v2
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v46, v6
	v_dual_mov_b32 v21, v5 :: v_dual_mov_b32 v44, v4
	v_dual_mov_b32 v19, v3 :: v_dual_mov_b32 v42, v2
	v_mov_b32_e32 v17, v1
	v_dual_mov_b32 v23, v7 :: v_dual_mov_b32 v22, v6
	v_dual_mov_b32 v53, v5 :: v_dual_mov_b32 v20, v4
	v_dual_mov_b32 v51, v3 :: v_dual_mov_b32 v18, v2
	v_mov_b32_e32 v49, v1
	v_dual_mov_b32 v55, v7 :: v_dual_mov_b32 v54, v6
	v_dual_mov_b32 v29, v5 :: v_dual_mov_b32 v52, v4
	v_dual_mov_b32 v27, v3 :: v_dual_mov_b32 v50, v2
	v_mov_b32_e32 v25, v1
	v_dual_mov_b32 v31, v7 :: v_dual_mov_b32 v30, v6
	v_dual_mov_b32 v61, v5 :: v_dual_mov_b32 v28, v4
	v_dual_mov_b32 v59, v3 :: v_dual_mov_b32 v26, v2
	v_mov_b32_e32 v57, v1
	v_dual_mov_b32 v63, v7 :: v_dual_mov_b32 v62, v6
	v_mov_b32_e32 v60, v4
	v_mov_b32_e32 v58, v2
	scratch_store_b64 off, v[0:1], off      ; 8-byte Folded Spill
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_8:                                ; %.preheader501.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s26, -1
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_mov_b32 s27, s9
	s_mov_b32 s29, s9
	s_branch .LBB3_10
.LBB3_9:                                ;   in Loop: Header=BB3_10 Depth=2
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v178, v223, v179
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v173, v173
	v_fmac_f32_e32 v64, v178, v177
	v_mul_f32_e32 v177, v222, v179
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v170, v170
	v_fmac_f32_e32 v63, v177, v176
	v_mul_f32_e32 v176, v221, v179
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_fmac_f32_e32 v62, v176, v175
	v_mul_f32_e32 v175, v220, v179
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v163, v163
	v_cvt_f32_i32_e32 v160, v160
	v_fmac_f32_e32 v61, v175, v174
	v_mul_f32_e32 v174, v219, v179
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v155, v155
	v_fmac_f32_e32 v60, v174, v173
	v_mul_f32_e32 v173, v218, v179
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v156, v156
	v_cvt_f32_i32_e32 v151, v151
	v_fmac_f32_e32 v59, v173, v172
	v_mul_f32_e32 v172, v217, v179
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v150, v150
	v_dual_fmac_f32 v58, v172, v171 :: v_dual_mul_f32 v171, v223, v180
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v147, v147
	v_fmac_f32_e32 v56, v171, v170
	v_mul_f32_e32 v170, v222, v180
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v143, v143
	v_fmac_f32_e32 v55, v170, v169
	v_mul_f32_e32 v169, v221, v180
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v140, v140
	v_cvt_f32_i32_e32 v134, v134
	v_fmac_f32_e32 v54, v169, v168
	v_mul_f32_e32 v168, v220, v180
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v130, v130
	v_fmac_f32_e32 v53, v168, v167
	v_mul_f32_e32 v167, v219, v180
	v_cvt_f32_i32_e32 v135, v135
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v131, v131
	v_fmac_f32_e32 v52, v167, v166
	v_mul_f32_e32 v166, v218, v180
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v127, v127
	v_fmac_f32_e32 v51, v166, v165
	v_mul_f32_e32 v165, v217, v180
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v117, v117
	v_fmac_f32_e32 v50, v165, v164
	v_mul_f32_e32 v164, v223, v250
	v_cvt_f32_i32_e32 v118, v118
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v113, v113
	v_cvt_f32_i32_e32 v121, v121
	v_dual_fmac_f32 v48, v164, v163 :: v_dual_mul_f32 v163, v222, v250
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v47, v163, v162
	v_mul_f32_e32 v162, v221, v250
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v111, v111
	v_cvt_f32_i32_e32 v112, v112
	v_dual_fmac_f32 v46, v162, v161 :: v_dual_mul_f32 v161, v220, v250
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v104, v104
	v_dual_fmac_f32 v45, v161, v160 :: v_dual_mul_f32 v160, v219, v250
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v103, v103
	v_cvt_f32_i32_e32 v105, v105
	v_dual_fmac_f32 v44, v160, v159 :: v_dual_mul_f32 v159, v218, v250
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v88, v88
	v_fmac_f32_e32 v43, v159, v158
	v_mul_f32_e32 v158, v217, v250
	v_cvt_f32_i32_e32 v90, v90
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v96, v96
	v_dual_fmac_f32 v42, v158, v157 :: v_dual_mul_f32 v157, v223, v248
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v83, v83
	v_fmac_f32_e32 v40, v157, v156
	v_mul_f32_e32 v156, v222, v248
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v74, v74
	v_fmac_f32_e32 v39, v156, v155
	v_mul_f32_e32 v155, v221, v248
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v85, v85
	v_fmac_f32_e32 v38, v155, v154
	v_mul_f32_e32 v154, v220, v248
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v76, v76
	v_cvt_f32_i32_e32 v78, v78
	v_fmac_f32_e32 v37, v154, v153
	v_mul_f32_e32 v153, v219, v248
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v69, v69
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v36, v153, v152
	v_mul_f32_e32 v152, v218, v248
	v_cvt_f32_i32_e32 v67, v67
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v86, v86
	v_cvt_f32_i32_e32 v73, v73
	v_fmac_f32_e32 v35, v152, v151
	v_mul_f32_e32 v151, v217, v248
	v_cvt_f32_i32_e32 v75, v75
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v70, v70
	v_fmac_f32_e32 v34, v151, v150
	v_mul_f32_e32 v150, v179, v231
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v66, v66
	v_cvt_f32_i32_e32 v68, v68
	s_xor_b32 s8, s26, -1
	v_dual_fmac_f32 v32, v150, v149 :: v_dual_mul_f32 v149, v179, v230
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s27, -1
	v_dual_fmac_f32 v31, v149, v148 :: v_dual_mul_f32 v148, v179, v227
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v148, v147 :: v_dual_mul_f32 v147, v179, v226
	v_fmac_f32_e32 v29, v147, v146
	v_mul_f32_e32 v146, v179, v225
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v146, v145 :: v_dual_mul_f32 v145, v179, v224
	v_dual_fmac_f32 v27, v145, v144 :: v_dual_mul_f32 v144, v179, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v144, v143
	v_mul_f32_e32 v143, v180, v231
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v142, v180, v230
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v142, v141
	v_mul_f32_e32 v141, v180, v227
	v_fmac_f32_e32 v22, v141, v140
	v_mul_f32_e32 v140, v180, v226
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v140, v139
	v_mul_f32_e32 v139, v180, v225
	v_fmac_f32_e32 v20, v139, v138
	v_mul_f32_e32 v138, v180, v224
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v138, v137
	v_mul_f32_e32 v137, v180, v229
	v_fmac_f32_e32 v18, v137, v136
	v_mul_f32_e32 v136, v250, v231
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v16, v136, v135 :: v_dual_mul_f32 v135, v250, v230
	v_dual_fmac_f32 v15, v135, v134 :: v_dual_mul_f32 v134, v250, v227
	v_mul_f32_e32 v135, v248, v225
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v134, v133
	v_mul_f32_e32 v133, v250, v226
	v_dual_fmac_f32 v13, v133, v132 :: v_dual_mul_f32 v132, v250, v225
	v_mul_f32_e32 v133, v248, v227
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v12, v132, v131 :: v_dual_mul_f32 v131, v250, v224
	v_mul_f32_e32 v132, v248, v226
	v_fmac_f32_e32 v6, v133, v126
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v11, v131, v130 :: v_dual_mul_f32 v130, v250, v229
	v_dual_mul_f32 v131, v248, v231 :: v_dual_fmac_f32 v10, v130, v129
	v_mul_f32_e32 v129, v248, v229
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v129, v122
	v_mul_f32_e32 v122, v239, v228
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v236, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v59, v115, v116
	v_mul_f32_e32 v115, v237, v228
	v_fmac_f32_e32 v5, v132, v125
	v_dual_fmac_f32 v60, v115, v117 :: v_dual_mul_f32 v115, v234, v228
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v134, v248, v224 :: v_dual_fmac_f32 v61, v115, v118
	v_dual_mul_f32 v115, v235, v228 :: v_dual_mul_f32 v130, v248, v230
	v_fmac_f32_e32 v3, v134, v123
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v62, v115, v119 :: v_dual_mul_f32 v115, v232, v228
	v_fmac_f32_e32 v7, v130, v127
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v63, v115, v120
	v_mul_f32_e32 v115, v233, v228
	v_fmac_f32_e32 v64, v115, v121
	v_mul_f32_e32 v115, v239, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v236, v0
	v_dual_fmac_f32 v8, v131, v128 :: v_dual_fmac_f32 v51, v108, v109
	v_mul_f32_e32 v108, v237, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v108, v110
	v_mul_f32_e32 v108, v234, v0
	v_dual_fmac_f32 v4, v135, v124 :: v_dual_fmac_f32 v53, v108, v111
	v_mul_f32_e32 v108, v235, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v54, v108, v112
	v_mul_f32_e32 v108, v232, v0
	v_dual_fmac_f32 v55, v108, v113 :: v_dual_mul_f32 v108, v233, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v56, v108, v114
	v_mul_f32_e32 v108, v239, v255
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v236, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v43, v101, v102
	v_mul_f32_e32 v101, v237, v255
	v_fmac_f32_e32 v44, v101, v103
	v_mul_f32_e32 v101, v234, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v45, v101, v104
	v_mul_f32_e32 v101, v235, v255
	v_dual_fmac_f32 v46, v101, v105 :: v_dual_mul_f32 v101, v232, v255
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v47, v101, v106
	v_mul_f32_e32 v101, v233, v255
	v_dual_fmac_f32 v48, v101, v107 :: v_dual_mul_f32 v101, v239, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v236, v181
	v_dual_fmac_f32 v35, v94, v95 :: v_dual_mul_f32 v94, v237, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v36, v94, v96
	v_mul_f32_e32 v94, v234, v181
	v_fmac_f32_e32 v37, v94, v97
	v_mul_f32_e32 v94, v235, v181
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v38, v94, v98
	v_mul_f32_e32 v94, v232, v181
	scratch_load_b64 v[231:232], off, off   ; 8-byte Folded Reload
	v_dual_fmac_f32 v39, v94, v99 :: v_dual_mul_f32 v94, v233, v181
	v_fmac_f32_e32 v40, v94, v100
	v_mul_f32_e32 v94, v228, v245
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v26, v94, v87 :: v_dual_mul_f32 v87, v228, v242
	v_fmac_f32_e32 v27, v87, v88
	v_mul_f32_e32 v87, v228, v243
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, v87, v89 :: v_dual_mul_f32 v87, v228, v240
	v_fmac_f32_e32 v29, v87, v90
	v_mul_f32_e32 v87, v228, v241
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v30, v87, v91 :: v_dual_mul_f32 v87, v228, v246
	v_fmac_f32_e32 v31, v87, v92
	v_mul_f32_e32 v87, v228, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v32, v87, v93
	v_mul_f32_e32 v87, v0, v245
	v_fmac_f32_e32 v18, v87, v80
	v_mul_f32_e32 v80, v0, v242
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v19, v80, v81
	v_mul_f32_e32 v80, v0, v243
	v_cvt_f32_i32_e32 v81, v190
	v_fmac_f32_e32 v20, v80, v82
	v_mul_f32_e32 v80, v0, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v21, v80, v83
	v_mul_f32_e32 v80, v0, v241
	v_fmac_f32_e32 v22, v80, v84
	v_mul_f32_e32 v80, v0, v246
	v_mul_f32_e32 v0, v0, v247
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v0, v86
	v_mul_f32_e32 v0, v255, v245
	v_fmac_f32_e32 v10, v0, v73
	v_mul_f32_e32 v0, v255, v242
	v_mul_f32_e32 v73, v181, v246
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v11, v0, v74 :: v_dual_mul_f32 v0, v255, v243
	v_fmac_f32_e32 v7, v73, v71
	v_mul_f32_e32 v71, v215, v194
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v73, v215, v192 :: v_dual_fmac_f32 v12, v0, v75
	v_dual_mul_f32 v0, v255, v240 :: v_dual_mul_f32 v75, v181, v242
	v_dual_fmac_f32 v13, v0, v76 :: v_dual_mul_f32 v0, v255, v241
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v3, v75, v67
	v_mul_f32_e32 v75, v189, v200
	v_dual_mul_f32 v67, v215, v198 :: v_dual_fmac_f32 v14, v0, v77
	v_dual_mul_f32 v0, v255, v246 :: v_dual_mul_f32 v77, v181, v240
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v15, v0, v78 :: v_dual_mul_f32 v0, v255, v247
	v_mul_f32_e32 v78, v181, v241
	v_fmac_f32_e32 v5, v77, v69
	v_mul_f32_e32 v77, v189, v194
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v69, v215, v200 :: v_dual_fmac_f32 v6, v78, v70
	v_mul_f32_e32 v70, v215, v199
	v_fmac_f32_e32 v16, v0, v79
	v_mul_f32_e32 v0, v181, v247
	v_dual_mul_f32 v78, v189, v193 :: v_dual_mul_f32 v79, v189, v191
	v_fmac_f32_e32 v8, v0, v72
	v_cvt_f32_i32_e32 v0, v216
	v_mul_f32_e32 v72, v215, v193
	v_mul_f32_e32 v74, v181, v245
	v_dual_fmac_f32 v5, v78, v81 :: v_dual_mul_f32 v76, v181, v243
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v31, v70, v0
	v_dual_fmac_f32 v29, v72, v0 :: v_dual_mul_f32 v72, v213, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v2, v74, v66
	v_mul_f32_e32 v66, v210, v215
	v_mul_f32_e32 v70, v213, v199
	v_fmac_f32_e32 v4, v76, v68
	v_mul_f32_e32 v68, v215, v191
	v_fmac_f32_e32 v32, v69, v0
	v_fmac_f32_e32 v64, v66, v0
	v_mul_f32_e32 v66, v209, v215
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v73, v0 :: v_dual_fmac_f32 v27, v68, v0
	v_fmac_f32_e32 v26, v67, v0
	v_dual_mul_f32 v76, v189, v199 :: v_dual_fmac_f32 v63, v66, v0
	v_dual_mul_f32 v66, v206, v215 :: v_dual_mul_f32 v67, v213, v198
	v_dual_mul_f32 v68, v213, v191 :: v_dual_mul_f32 v69, v213, v200
	v_mul_f32_e32 v73, v213, v192
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v62, v66, v0
	v_mul_f32_e32 v66, v205, v215
	v_dual_mul_f32 v74, v189, v198 :: v_dual_fmac_f32 v7, v76, v81
	v_fmac_f32_e32 v3, v79, v81
	v_dual_fmac_f32 v61, v66, v0 :: v_dual_mul_f32 v66, v204, v215
	v_dual_fmac_f32 v23, v80, v85 :: v_dual_fmac_f32 v30, v71, v0
	v_mul_f32_e32 v71, v213, v194
	v_mul_f32_e32 v80, v189, v192
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v60, v66, v0
	v_mul_f32_e32 v66, v203, v215
	v_dual_fmac_f32 v59, v66, v0 :: v_dual_mul_f32 v66, v208, v215
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v58, v66, v0
	v_mul_f32_e32 v66, v207, v215
	v_dual_fmac_f32 v57, v66, v0 :: v_dual_mul_f32 v66, v215, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v66, v0
	v_cvt_f32_i32_e32 v0, v214
	v_mul_f32_e32 v66, v210, v213
	v_fmac_f32_e32 v24, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v56, v66, v0
	v_dual_mul_f32 v66, v209, v213 :: v_dual_fmac_f32 v23, v70, v0
	v_dual_fmac_f32 v22, v71, v0 :: v_dual_fmac_f32 v21, v72, v0
	v_dual_fmac_f32 v20, v73, v0 :: v_dual_fmac_f32 v55, v66, v0
	v_dual_mul_f32 v66, v206, v213 :: v_dual_fmac_f32 v19, v68, v0
	v_fmac_f32_e32 v18, v67, v0
	v_dual_mul_f32 v67, v211, v198 :: v_dual_mul_f32 v68, v211, v191
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v54, v66, v0
	v_dual_mul_f32 v66, v205, v213 :: v_dual_mul_f32 v69, v211, v200
	v_dual_mul_f32 v70, v211, v199 :: v_dual_mul_f32 v71, v211, v194
	v_dual_mul_f32 v72, v211, v193 :: v_dual_fmac_f32 v53, v66, v0
	v_dual_mul_f32 v66, v204, v213 :: v_dual_mul_f32 v73, v211, v192
	v_fmac_f32_e32 v6, v77, v81
	v_fmac_f32_e32 v4, v80, v81
	v_fmac_f32_e32 v2, v74, v81
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v52, v66, v0
	v_mul_f32_e32 v66, v203, v213
	v_dual_fmac_f32 v51, v66, v0 :: v_dual_mul_f32 v66, v208, v213
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v50, v66, v0
	v_mul_f32_e32 v66, v207, v213
	v_dual_fmac_f32 v49, v66, v0 :: v_dual_mul_f32 v66, v213, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v66, v0
	v_cvt_f32_i32_e32 v0, v212
	v_mul_f32_e32 v66, v210, v211
	v_fmac_f32_e32 v16, v69, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v48, v66, v0
	v_dual_mul_f32 v66, v209, v211 :: v_dual_fmac_f32 v15, v70, v0
	v_dual_fmac_f32 v14, v71, v0 :: v_dual_fmac_f32 v13, v72, v0
	v_dual_fmac_f32 v12, v73, v0 :: v_dual_fmac_f32 v47, v66, v0
	v_dual_mul_f32 v66, v206, v211 :: v_dual_fmac_f32 v11, v68, v0
	v_dual_fmac_f32 v10, v67, v0 :: v_dual_mul_f32 v67, v210, v189
	v_mul_f32_e32 v68, v209, v189
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v46, v66, v0
	v_dual_mul_f32 v66, v205, v211 :: v_dual_mul_f32 v69, v206, v189
	v_dual_mul_f32 v70, v205, v189 :: v_dual_mul_f32 v71, v203, v189
	v_dual_mul_f32 v72, v204, v189 :: v_dual_fmac_f32 v45, v66, v0
	v_dual_mul_f32 v66, v204, v211 :: v_dual_mul_f32 v73, v189, v197
	v_dual_fmac_f32 v40, v67, v81 :: v_dual_fmac_f32 v39, v68, v81
	v_fmac_f32_e32 v38, v69, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v66, v0
	v_dual_mul_f32 v66, v203, v211 :: v_dual_fmac_f32 v37, v70, v81
	v_dual_fmac_f32 v36, v72, v81 :: v_dual_fmac_f32 v35, v71, v81
	v_dual_fmac_f32 v8, v75, v81 :: v_dual_fmac_f32 v43, v66, v0
	v_dual_mul_f32 v66, v208, v211 :: v_dual_fmac_f32 v1, v73, v81
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v42, v66, v0
	v_mul_f32_e32 v66, v207, v211
	v_dual_fmac_f32 v41, v66, v0 :: v_dual_mul_f32 v66, v211, v197
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v9, v66, v0 :: v_dual_mul_f32 v0, v207, v189
	v_mul_f32_e32 v66, v208, v189
	v_dual_fmac_f32 v33, v0, v81 :: v_dual_fmac_f32 v34, v66, v81
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_7
.LBB3_10:                               ;   Parent Loop BB3_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	scratch_load_b32 v0, off, off offset:8  ; 4-byte Folded Reload
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s29, s11
	s_and_b32 s30, s26, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[8:9], s[14:15]
	s_cselect_b32 s33, s13, 0x3400
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_cselect_b32 s36, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_mov_b32 s35, s9
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v66, s34, s30, v231
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v67, null, s31, 0, s34
	s_lshl_b32 s34, s29, 6
	v_add_co_u32 v68, s30, s30, v183
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	v_add_co_ci_u32_e64 v69, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v70, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v71, null, s35, 0, s30
	v_add_co_u32 v72, s30, s34, v184
	s_clause 0x1
	global_load_b64 v[66:67], v[66:67], off offset:40
	global_load_b64 v[68:69], v[68:69], off offset:40
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s30
	s_clause 0x1
	global_load_b64 v[185:186], v[70:71], off offset:40
	global_load_b64 v[187:188], v[72:73], off offset:40
	v_add_nc_u32_e32 v76, s28, v253
	ds_load_b64 v[70:71], v76
	s_wait_dscnt 0x0
	v_dual_mov_b32 v75, v71 :: v_dual_mov_b32 v74, v70
	s_wait_loadcnt 0x4
	v_add_nc_u32_e32 v249, s36, v0
	ds_load_b64 v[72:73], v254
	ds_load_2addr_b32 v[215:216], v249 offset1:2
	ds_load_2addr_b32 v[217:218], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[219:220], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[221:222], v249 offset0:12 offset1:14
	;;#ASMSTART
	v_xor_b32 v74, v74, v65
	;;#ASMEND
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[170:177], v[74:75], v[72:73], 0 neg_lo:[0,1,0]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v196, 0, v67, s0
	v_cndmask_b32_e64 v195, 0, v66, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v202, 0, v69, s1
	v_cndmask_b32_e64 v201, 0, v68, s1
	; sched_barrier mask(0x00000000)
	scratch_load_b32 v0, off, off offset:12 ; 4-byte Folded Reload
	v_cvt_f32_i32_e32 v65, v170
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v178, s33, v0
	ds_load_b32 v226, v178
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v226
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v57, v0, v65, v57
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v70 :: v_dual_mov_b32 v68, v71
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v225, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v225
	v_fma_f32 v49, v0, v65, v49
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v70
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v224, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v224
	v_fma_f32 v41, v0, v65, v41
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v70, v70, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[70:71], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v223, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v215, v223
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, v0, v65, v33
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:512
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[208:209], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[210:211], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[212:213], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[214:215], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v226, v208
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, v0, v67, v25
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v225, v208
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v17, v0, v67, v17
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v224, v208
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, v0, v67, v9
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v223, v208
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fma_f32 v1, v0, v65, v1
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v76 offset:256
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[207:208], v249 offset1:2
	ds_load_2addr_b32 v[205:206], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[203:204], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[199:200], v249 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v181
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:128
	v_cvt_f32_i32_e32 v67, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v180
	v_fmac_f32_e32 v49, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178 offset:256
	v_cvt_f32_i32_e32 v67, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v207, v179
	v_fmac_f32_e32 v41, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v207, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[227:228], v76 offset:768
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[197:198], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[193:194], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[191:192], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[189:190], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v227 :: v_dual_mov_b32 v68, v228
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v181, v197
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v180, v197
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v227
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v179, v197
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[229:230], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v227, v227, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[227:228], v[229:230], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v197, v0, v197
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v197, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s28, s27, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s28
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:16
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.11:                               ;   in Loop: Header=BB3_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v195, off, off offset:36
	scratch_load_b32 v197, off, off offset:48
	s_add_co_i32 s8, s8, 1
	s_mov_b32 s37, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[8:9], s[14:15]
	s_add_co_i32 s8, s29, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_mul_u64 s[34:35], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_xor_b32 s29, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v185, s33, s30, v231
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v186, null, s31, 0, s33
	v_add_co_u32 v187, s33, s30, v183
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v188, null, s31, 0, s33
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[195:196], null, 0x48, v195, s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v227, s30, s34, v182
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v228, null, s35, 0, s30
	v_add_co_u32 v229, s30, s34, v184
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v230, null, s35, 0, s30
	s_wait_loadcnt 0x0
	v_add_co_u32 v231, vcc_lo, v195, v197
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v232, null, 0, v196, vcc_lo
	scratch_load_b96 v[195:197], off, off offset:24 ; 12-byte Folded Reload
	s_wait_loadcnt 0x0
	v_add_co_u32 v195, vcc_lo, v195, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v196, null, 0, v196, vcc_lo
	s_lshl_b32 s8, s29, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v233, vcc_lo, v195, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v234, null, 0, v196, vcc_lo
	s_clause 0x1
	global_load_b64 v[195:196], v[185:186], off offset:8
	global_load_b64 v[201:202], v[187:188], off offset:8
	s_clause 0x1
	global_load_b64 v[185:186], v[227:228], off offset:8
	global_load_b64 v[187:188], v[229:230], off offset:8
	global_load_b32 v197, v[231:232], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:16 ; 4-byte Folded Spill
	global_load_b32 v197, v[233:234], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v197, off offset:20 ; 4-byte Folded Spill
.LBB3_12:                               ; %.preheader499.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_dual_mul_f32 v197, v222, v226 :: v_dual_mul_f32 v228, v219, v226
	v_mul_f32_e32 v207, v221, v226
	v_dual_mul_f32 v227, v220, v226 :: v_dual_mul_f32 v230, v217, v226
	v_mul_f32_e32 v231, v216, v226
	v_cvt_f32_i32_e32 v177, v177
	v_mul_f32_e32 v229, v218, v226
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_fmac_f32_e32 v64, v197, v177
	v_fma_f32 v59, v230, v172, v59
	v_fma_f32 v60, v229, v173, v60
	v_fma_f32 v61, v228, v174, v61
	v_fma_f32 v62, v227, v175, v62
	v_mul_f32_e32 v172, v221, v225
	v_fma_f32 v63, v207, v176, v63
	v_mul_f32_e32 v174, v219, v225
	v_fma_f32 v58, v231, v171, v58
	v_dual_mul_f32 v171, v222, v225 :: v_dual_mul_f32 v176, v217, v225
	v_mul_f32_e32 v175, v218, v225
	v_mul_f32_e32 v177, v216, v225
	v_cvt_f32_i32_e32 v170, v170
	v_mul_f32_e32 v173, v220, v225
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_fma_f32 v50, v177, v164, v50
	v_fma_f32 v51, v176, v165, v51
	v_fma_f32 v52, v175, v166, v52
	v_fma_f32 v53, v174, v167, v53
	v_fma_f32 v55, v172, v169, v55
	v_fmac_f32_e32 v56, v171, v170
	v_fma_f32 v54, v173, v168, v54
	v_dual_mul_f32 v164, v222, v224 :: v_dual_mul_f32 v165, v221, v224
	v_mul_f32_e32 v168, v218, v224
	v_dual_mul_f32 v166, v220, v224 :: v_dual_mul_f32 v167, v219, v224
	v_mul_f32_e32 v170, v216, v224
	v_cvt_f32_i32_e32 v163, v163
	v_mul_f32_e32 v169, v217, v224
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v162, v162
	v_fma_f32 v42, v170, v157, v42
	v_fma_f32 v44, v168, v159, v44
	v_fma_f32 v45, v167, v160, v45
	v_fma_f32 v46, v166, v161, v46
	v_fmac_f32_e32 v48, v164, v163
	v_fma_f32 v43, v169, v158, v43
	v_dual_mul_f32 v157, v222, v223 :: v_dual_mul_f32 v158, v221, v223
	v_dual_mul_f32 v159, v220, v223 :: v_dual_mul_f32 v160, v219, v223
	v_mul_f32_e32 v161, v218, v223
	v_cvt_f32_i32_e32 v156, v156
	v_mul_f32_e32 v163, v216, v223
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v153, v153
	v_fma_f32 v47, v165, v162, v47
	v_mul_f32_e32 v162, v217, v223
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v154, v154
	v_fma_f32 v36, v161, v152, v36
	v_fma_f32 v37, v160, v153, v37
	v_fmac_f32_e32 v40, v157, v156
	v_fma_f32 v34, v163, v150, v34
	v_mul_f32_e32 v150, v226, v215
	v_dual_mul_f32 v152, v226, v213 :: v_dual_mul_f32 v153, v226, v212
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v146, v146
	v_fma_f32 v35, v162, v151, v35
	v_fma_f32 v38, v159, v154, v38
	v_fma_f32 v39, v158, v155, v39
	v_dual_mul_f32 v151, v226, v214 :: v_dual_mul_f32 v154, v226, v211
	v_mul_f32_e32 v155, v226, v210
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v147, v147
	v_fma_f32 v29, v153, v146, v29
	v_fmac_f32_e32 v32, v150, v149
	v_dual_mul_f32 v146, v225, v212 :: v_dual_mul_f32 v149, v225, v209
	v_cvt_f32_i32_e32 v136, v136
	v_mul_f32_e32 v156, v226, v209
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v145, v145
	v_fma_f32 v27, v155, v144, v27
	v_fma_f32 v30, v152, v147, v30
	v_fma_f32 v31, v151, v148, v31
	v_dual_mul_f32 v144, v225, v214 :: v_dual_mul_f32 v147, v225, v211
	v_mul_f32_e32 v148, v225, v210
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_fma_f32 v18, v149, v136, v18
	v_mul_f32_e32 v136, v224, v215
	v_cvt_f32_i32_e32 v135, v135
	v_fma_f32 v26, v156, v143, v26
	v_fma_f32 v28, v154, v145, v28
	v_mul_f32_e32 v143, v225, v215
	v_mul_f32_e32 v145, v225, v213
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v140, v140
	v_fma_f32 v20, v147, v138, v20
	v_fma_f32 v21, v146, v139, v21
	v_fma_f32 v23, v144, v141, v23
	v_dual_mul_f32 v138, v224, v213 :: v_dual_mul_f32 v141, v224, v210
	v_mul_f32_e32 v139, v224, v212
	v_fma_f32 v19, v148, v137, v19
	v_mul_f32_e32 v137, v224, v214
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v130, v130
	v_dual_fmac_f32 v16, v136, v135 :: v_dual_mul_f32 v135, v223, v209
	v_cvt_f32_i32_e32 v122, v122
	v_fma_f32 v22, v145, v140, v22
	v_fmac_f32_e32 v24, v143, v142
	v_mul_f32_e32 v140, v224, v211
	v_mul_f32_e32 v142, v224, v209
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_fma_f32 v11, v141, v130, v11
	v_fma_f32 v15, v137, v134, v15
	v_mul_f32_e32 v130, v223, v214
	v_mul_f32_e32 v134, v223, v210
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v123, v123
	v_fma_f32 v2, v135, v122, v2
	v_mul_f32_e32 v122, v208, v181
	v_cvt_f32_i32_e32 v115, v115
	v_fma_f32 v10, v142, v129, v10
	v_fma_f32 v12, v140, v131, v12
	v_fma_f32 v13, v139, v132, v13
	v_fma_f32 v14, v138, v133, v14
	v_mul_f32_e32 v129, v223, v215
	v_mul_f32_e32 v131, v223, v213
	v_mul_f32_e32 v133, v223, v211
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v126, v126
	v_fma_f32 v7, v130, v127, v7
	v_mul_f32_e32 v127, v199, v181
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v118, v118
	v_mul_f32_e32 v132, v223, v212
	v_dual_fmac_f32 v58, v122, v115 :: v_dual_mul_f32 v115, v208, v180
	v_fma_f32 v3, v134, v123, v3
	v_mul_f32_e32 v123, v205, v181
	v_fma_f32 v4, v133, v124, v4
	v_fma_f32 v5, v132, v125, v5
	v_fma_f32 v6, v131, v126, v6
	v_dual_fmac_f32 v8, v129, v128 :: v_dual_mul_f32 v125, v203, v181
	v_dual_mul_f32 v124, v206, v181 :: v_dual_fmac_f32 v63, v127, v120
	v_dual_mul_f32 v126, v204, v181 :: v_dual_fmac_f32 v59, v123, v116
	v_mul_f32_e32 v128, v200, v181
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v117, v117
	v_mul_f32_e32 v120, v199, v180
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v113, v113
	v_mul_f32_e32 v116, v205, v180
	v_dual_fmac_f32 v62, v126, v119 :: v_dual_mul_f32 v119, v204, v180
	v_dual_fmac_f32 v61, v125, v118 :: v_dual_fmac_f32 v60, v124, v117
	v_mul_f32_e32 v117, v206, v180
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v111, v111
	v_dual_fmac_f32 v55, v120, v113 :: v_dual_mul_f32 v118, v203, v180
	v_fmac_f32_e32 v51, v116, v109
	v_fmac_f32_e32 v64, v128, v121
	v_mul_f32_e32 v121, v200, v180
	v_mul_f32_e32 v113, v199, v179
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v110, v110
	v_fmac_f32_e32 v56, v121, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v54, v119, v112 :: v_dual_fmac_f32 v47, v113, v106
	v_fmac_f32_e32 v53, v118, v111
	v_dual_mul_f32 v111, v203, v179 :: v_dual_mul_f32 v112, v204, v179
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v104, v104
	v_mul_f32_e32 v106, v199, v0
	v_cvt_f32_i32_e32 v99, v99
	v_fmac_f32_e32 v52, v117, v110
	v_dual_mul_f32 v109, v205, v179 :: v_dual_mul_f32 v110, v206, v179
	v_mul_f32_e32 v114, v200, v179
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_fmac_f32_e32 v50, v115, v108
	v_mul_f32_e32 v108, v208, v179
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v103, v103
	v_dual_fmac_f32 v46, v112, v105 :: v_dual_fmac_f32 v39, v106, v99
	v_fmac_f32_e32 v45, v111, v104
	v_dual_mul_f32 v104, v203, v0 :: v_dual_mul_f32 v105, v204, v0
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v97, v97
	v_mul_f32_e32 v99, v181, v189
	v_cvt_f32_i32_e32 v92, v92
	v_fmac_f32_e32 v48, v114, v107
	v_dual_fmac_f32 v44, v110, v103 :: v_dual_fmac_f32 v43, v109, v102
	v_dual_mul_f32 v102, v205, v0 :: v_dual_mul_f32 v103, v206, v0
	v_mul_f32_e32 v107, v200, v0
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v95, v95
	v_fmac_f32_e32 v42, v108, v101
	v_mul_f32_e32 v101, v208, v0
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v96, v96
	v_dual_fmac_f32 v38, v105, v98 :: v_dual_fmac_f32 v31, v99, v92
	v_fmac_f32_e32 v37, v104, v97
	v_mul_f32_e32 v97, v181, v191
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v74, v74
	v_mul_f32_e32 v92, v180, v189
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v40, v107, v100 :: v_dual_fmac_f32 v29, v97, v90
	v_fmac_f32_e32 v36, v103, v96
	v_dual_mul_f32 v96, v181, v194 :: v_dual_fmac_f32 v23, v92, v85
	v_mul_f32_e32 v98, v181, v192
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v88, v88
	v_fmac_f32_e32 v34, v101, v94
	v_mul_f32_e32 v94, v181, v198
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v91, v91
	v_dual_mul_f32 v90, v180, v191 :: v_dual_mul_f32 v85, v179, v189
	v_cvt_f32_i32_e32 v83, v83
	v_mul_f32_e32 v100, v181, v190
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v82, v82
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v21, v90, v83 :: v_dual_fmac_f32 v32, v100, v93
	v_mul_f32_e32 v83, v179, v191
	v_dual_mul_f32 v93, v180, v190 :: v_dual_fmac_f32 v30, v98, v91
	v_dual_mul_f32 v91, v180, v192 :: v_dual_fmac_f32 v26, v94, v87
	v_mul_f32_e32 v87, v180, v198
	v_fmac_f32_e32 v35, v102, v95
	v_mul_f32_e32 v95, v181, v193
	v_cvt_f32_i32_e32 v78, v78
	v_fmac_f32_e32 v28, v96, v89
	v_mul_f32_e32 v89, v180, v194
	v_cvt_f32_i32_e32 v76, v76
	v_dual_fmac_f32 v27, v95, v88 :: v_dual_mul_f32 v88, v180, v193
	v_cvt_f32_i32_e32 v80, v80
	v_cvt_f32_i32_e32 v86, v86
	v_fmac_f32_e32 v20, v89, v82
	v_mul_f32_e32 v82, v179, v194
	v_cvt_f32_i32_e32 v75, v75
	v_fmac_f32_e32 v13, v83, v76
	v_fmac_f32_e32 v19, v88, v81
	v_dual_mul_f32 v81, v179, v193 :: v_dual_fmac_f32 v24, v93, v86
	v_fmac_f32_e32 v18, v87, v80
	v_cvt_f32_i32_e32 v73, v73
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v12, v82, v75 :: v_dual_fmac_f32 v11, v81, v74
	v_add_nc_u32_e32 v75, 0, v253
	v_mul_f32_e32 v81, v0, v189
	v_cvt_f32_i32_e32 v71, v71
	v_mul_f32_e32 v80, v179, v198
	v_cvt_f32_i32_e32 v82, v66
	v_cvt_f32_i32_e32 v83, v67
	ds_load_b64 v[66:67], v254
	v_dual_fmac_f32 v7, v81, v71 :: v_dual_fmac_f32 v10, v80, v73
	ds_load_b64 v[73:74], v75 offset:8192
	ds_load_2addr_b32 v[216:217], v249 offset1:2
	ds_load_2addr_b32 v[218:219], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[220:221], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[222:223], v249 offset0:12 offset1:14
	v_cvt_f32_i32_e32 v84, v84
	v_mul_f32_e32 v86, v179, v190
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v77, v77
	v_cvt_f32_i32_e32 v69, v69
	v_fmac_f32_e32 v22, v91, v84
	v_mul_f32_e32 v84, v179, v192
	v_fmac_f32_e32 v16, v86, v79
	v_mul_f32_e32 v79, v0, v191
	v_fmac_f32_e32 v15, v85, v78
	v_mul_f32_e32 v76, v0, v198
	v_fmac_f32_e32 v14, v84, v77
	v_dual_mul_f32 v77, v0, v193 :: v_dual_mul_f32 v78, v0, v194
	v_mul_f32_e32 v80, v0, v192
	v_mul_f32_e32 v0, v0, v190
	v_cvt_f32_i32_e32 v72, v72
	v_cvt_f32_i32_e32 v70, v70
	v_cvt_f32_i32_e32 v84, v68
	v_fmac_f32_e32 v5, v79, v69
	s_wait_dscnt 0x4
	v_dual_mov_b32 v69, v74 :: v_dual_mov_b32 v68, v73
	v_fmac_f32_e32 v8, v0, v72
	v_fmac_f32_e32 v6, v80, v70
	v_dual_fmac_f32 v4, v78, v84 :: v_dual_fmac_f32 v3, v77, v83
	v_fmac_f32_e32 v2, v76, v82
	;;#ASMSTART
	v_xor_b32 v68, v68, v65
	;;#ASMEND
	v_wmma_i32_16x16x32_iu4 v[170:177], v[68:69], v[66:67], 0 neg_lo:[0,1,0]
	s_xor_b32 s8, s28, -1
	; sched_barrier mask(0x00000000)
	ds_load_b32 v179, v178
	v_cvt_f32_i32_e32 v65, v170
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v179
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v57, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:512
	v_dual_mov_b32 v67, v73 :: v_dual_mov_b32 v68, v74
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[163:170], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v180, v178 offset:128
	v_cvt_f32_i32_e32 v65, v163
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v180
	v_fmac_f32_e32 v49, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1024
	v_mov_b32_e32 v67, v73
	;;#ASMSTART
	v_xor_b32 v67, v67, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[156:163], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v250, v178 offset:256
	v_cvt_f32_i32_e32 v65, v156
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v250
	v_fmac_f32_e32 v41, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v73, v73, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[149:156], v[73:74], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v248, v178 offset:384
	v_cvt_f32_i32_e32 v65, v149
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v216, v248
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8704
	ds_load_b64 v[67:68], v254
	ds_load_2addr_b32 v[228:229], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[224:225], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[226:227], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[230:231], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[142:149], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v0, v179, v228
	v_cvt_f32_i32_e32 v67, v142
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:512
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[135:142], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v180, v228
	v_cvt_f32_i32_e32 v67, v135
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1024
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[128:135], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v250, v228
	v_cvt_f32_i32_e32 v67, v128
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1536
	;;#ASMSTART
	v_xor_b32 v65, v65, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[121:128], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v0, v248, v228
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v121
	v_fmac_f32_e32 v1, v0, v65
	;;#ASMSTART
	v_xor_b32 v0, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v75 offset:8448
	ds_load_b64 v[67:68], v254 offset:256
	ds_load_2addr_b32 v[238:239], v249 offset1:2
	ds_load_2addr_b32 v[236:237], v249 offset0:4 offset1:6
	ds_load_2addr_b32 v[234:235], v249 offset0:8 offset1:10
	ds_load_2addr_b32 v[232:233], v249 offset0:12 offset1:14
	s_wait_dscnt 0x5
	v_dual_mov_b32 v69, v65 :: v_dual_mov_b32 v70, v66
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[114:121], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v228, v178
	v_cvt_f32_i32_e32 v67, v114
	s_wait_dscnt 0x0
	v_mul_f32_e32 v0, v238, v228
	v_fmac_f32_e32 v57, v0, v67
	;;#ASMSTART
	v_xor_b32 v0, v57, v57
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:768
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v0
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[107:114], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v0, v178 offset:128
	v_cvt_f32_i32_e32 v68, v107
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v0
	v_fmac_f32_e32 v49, v67, v68
	;;#ASMSTART
	v_xor_b32 v71, v49, v49
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1280
	v_mov_b32_e32 v69, v65
	;;#ASMSTART
	v_xor_b32 v69, v69, v71
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_i32_16x16x32_iu4 v[100:107], v[69:70], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v255, v178 offset:256
	v_cvt_f32_i32_e32 v68, v100
	s_wait_dscnt 0x0
	v_mul_f32_e32 v67, v238, v255
	v_fmac_f32_e32 v41, v67, v68
	;;#ASMSTART
	v_xor_b32 v69, v41, v41
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[67:68], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v65, v65, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[93:100], v[65:66], v[67:68], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_b32 v181, v178 offset:384
	v_cvt_f32_i32_e32 v66, v93
	s_wait_dscnt 0x0
	v_mul_f32_e32 v65, v238, v181
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v33, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v33, v33
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[189:190], v75 offset:8960
	ds_load_b64 v[65:66], v254 offset:256
	ds_load_2addr_b32 v[244:245], v249 offset0:32 offset1:34
	ds_load_2addr_b32 v[242:243], v249 offset0:36 offset1:38
	ds_load_2addr_b32 v[240:241], v249 offset0:40 offset1:42
	ds_load_2addr_b32 v[246:247], v249 offset0:44 offset1:46
	s_wait_dscnt 0x5
	v_dual_mov_b32 v67, v189 :: v_dual_mov_b32 v68, v190
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[86:93], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_dscnt 0x3
	v_mul_f32_e32 v65, v228, v244
	v_cvt_f32_i32_e32 v66, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v25, v25
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:768
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[79:86], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v0, v244
	v_cvt_f32_i32_e32 v66, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, v65, v66
	;;#ASMSTART
	v_xor_b32 v69, v17, v17
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[65:66], v254 offset:1280
	v_mov_b32_e32 v67, v189
	;;#ASMSTART
	v_xor_b32 v67, v67, v69
	;;#ASMEND
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_wmma_i32_16x16x32_iu4 v[72:79], v[67:68], v[65:66], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v65, v255, v244
	v_cvt_f32_i32_e32 v66, v72
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v65, v66
	;;#ASMSTART
	v_xor_b32 v65, v9, v9
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_load_b64 v[191:192], v254 offset:1792
	;;#ASMSTART
	v_xor_b32 v189, v189, v65
	;;#ASMEND
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[65:72], v[189:190], v[191:192], 0 neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	v_mul_f32_e32 v189, v181, v244
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_i32_e32 v65, v65
	v_fmac_f32_e32 v1, v189, v65
	;;#ASMSTART
	v_xor_b32 v65, v1, v1
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b32 v[207:208], v249 offset0:1 offset1:3
	ds_load_2addr_b32 v[203:204], v249 offset0:5 offset1:7
	ds_load_2addr_b32 v[205:206], v249 offset0:9 offset1:11
	ds_load_2addr_b32 v[209:210], v249 offset0:13 offset1:15
	ds_load_2addr_b32 v[215:216], v178 offset1:1
	ds_load_2addr_b32 v[213:214], v178 offset0:32 offset1:33
	ds_load_2addr_b32 v[211:212], v178 offset0:64 offset1:65
	ds_load_2addr_b32 v[189:190], v178 offset0:96 offset1:97
	ds_load_2addr_b32 v[197:198], v249 offset0:33 offset1:35
	ds_load_2addr_b32 v[191:192], v249 offset0:37 offset1:39
	ds_load_2addr_b32 v[193:194], v249 offset0:41 offset1:43
	ds_load_2addr_b32 v[199:200], v249 offset0:45 offset1:47
	s_movk_i32 s28, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_9
; %bb.13:                               ; %.preheader500.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v178, off, off offset:44
	scratch_load_b32 v238, off, off offset:20
	s_and_b32 s8, s27, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v196, 0, v196, s0
	v_cndmask_b32_e64 v195, 0, v195, s0
	v_cndmask_b32_e64 v202, 0, v202, s1
	v_cndmask_b32_e64 v201, 0, v201, s1
	s_movk_i32 s28, 0x1000
	ds_store_2addr_stride64_b64 v251, v[195:196], v[185:186] offset1:8
	ds_store_2addr_stride64_b64 v252, v[201:202], v[187:188] offset1:8
	scratch_load_b32 v249, off, off offset:40 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v178, v238
	scratch_load_b32 v238, off, off offset:16 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v178, v178.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_loadcnt 0x1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v244, s8, v249
	s_cselect_b32 s8, s23, 0x3c00
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v249, s8, v249
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v238, 0, v238, s2
	ds_store_b32 v244, v238
	ds_store_b32 v249, v178
	s_branch .LBB3_9
.LBB3_14:                               ; %.preheader497.i.loopexit
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v0, off, off offset:52
	scratch_load_b32 v67, off, off offset:56
	scratch_load_b32 v69, off, off offset:60
	s_wait_loadcnt 0x2
	v_bfe_u32 v68, v0, 4, 1
.LBB3_15:                               ; %.preheader497.i
	s_wait_loadcnt 0x0
	v_mul_u32_u24_e32 v66, 0x500, v69
	v_lshlrev_b32_e32 v65, 2, v67
	v_lshrrev_b32_e32 v0, 4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v65, 0, v66, v65
	v_and_b32_e32 v0, 15, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_u32_u24 v66, 0x280, v68, v65
	ds_store_2addr_b32 v66, v57, v58 offset1:20
	ds_store_2addr_b32 v66, v59, v60 offset0:40 offset1:60
	ds_store_2addr_b32 v66, v61, v62 offset0:80 offset1:100
	ds_store_2addr_b32 v66, v63, v64 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v57, s20, v67
	v_or_b32_e32 v60, s21, v0
	v_mul_u32_u24_e32 v59, 0x50, v67
	v_lshlrev_b32_e32 v0, 2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v57
	v_cmp_gt_i32_e32 vcc_lo, s14, v60
	v_ashrrev_i32_e32 v58, 31, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v0, 0, v59, v0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_17
; %bb.16:
	v_mad_co_i64_i32 v[61:62], null, s12, v60, 0
	ds_load_b32 v59, v0
	v_lshlrev_b64_e32 v[63:64], 2, v[57:58]
	v_lshlrev_b64_e32 v[61:62], 2, v[61:62]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v61, s0, s16, v61
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v62, null, s17, v62, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v61, s0, v61, v63
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v62, null, v62, v64, s0
	s_wait_dscnt 0x0
	global_store_b32 v[61:62], v59, off
.LBB3_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v61, 64, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v61
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_19
; %bb.18:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v59, v0 offset:1280
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off
.LBB3_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 32, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v59
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_21
; %bb.20:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v59, v0 offset:2560
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off offset:128
.LBB3_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_23
; %bb.22:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v59, v0 offset:3840
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off offset:128
.LBB3_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 64, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v59
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_25
; %bb.24:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v59, v0 offset:5120
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off offset:256
.LBB3_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_27
; %bb.26:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v59, v0 offset:6400
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off offset:256
.LBB3_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v59, 0x60, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v59
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_29
; %bb.28:
	v_mad_co_i64_i32 v[62:63], null, s12, v60, 0
	ds_load_b32 v59, v0 offset:7680
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v59, off offset:384
.LBB3_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v59, 0x280, v68
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_31
; %bb.30:
	v_mad_co_i64_i32 v[62:63], null, s12, v61, 0
	ds_load_b32 v64, v0 offset:8960
	v_lshlrev_b64_e32 v[66:67], 2, v[57:58]
	v_lshlrev_b64_e32 v[62:63], 2, v[62:63]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, s16, v62
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, s17, v63, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v62, s1, v62, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v63, null, v63, v67, s1
	s_wait_dscnt 0x0
	global_store_b32 v[62:63], v64, off offset:384
.LBB3_31:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v59, v65, v59
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v49, v50 offset1:20
	ds_store_2addr_b32 v59, v51, v52 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v53, v54 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v55, v56 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v49, 16, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v49
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_33
; %bb.32:
	v_mad_co_i64_i32 v[50:51], null, s12, v49, 0
	ds_load_b32 v54, v0
	v_lshlrev_b64_e32 v[52:53], 2, v[57:58]
	v_lshlrev_b64_e32 v[50:51], 2, v[50:51]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v50, s2, s16, v50
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v51, null, s17, v51, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v50, s2, v50, v52
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v51, null, v51, v53, s2
	s_wait_dscnt 0x0
	global_store_b32 v[50:51], v54, off
.LBB3_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v50, 0x50, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v50
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_106
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_107
.LBB3_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_108
.LBB3_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_109
.LBB3_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_110
.LBB3_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_111
.LBB3_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_41
.LBB3_40:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:8960
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:384
.LBB3_41:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v41, v42 offset1:20
	ds_store_2addr_b32 v59, v43, v44 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v45, v46 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v47, v48 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v41, 32, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v41
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_43
; %bb.42:
	v_mad_co_i64_i32 v[42:43], null, s12, v41, 0
	ds_load_b32 v46, v0
	v_lshlrev_b64_e32 v[44:45], 2, v[57:58]
	v_lshlrev_b64_e32 v[42:43], 2, v[42:43]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, s4, s16, v42
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v43, null, s17, v43, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v42, s4, v42, v44
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v43, null, v43, v45, s4
	s_wait_dscnt 0x0
	global_store_b32 v[42:43], v46, off
.LBB3_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v42, 0x60, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v42
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_112
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_113
.LBB3_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_114
.LBB3_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_115
.LBB3_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_116
.LBB3_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_117
.LBB3_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_51
.LBB3_50:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:8960
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:384
.LBB3_51:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v33, v34 offset1:20
	ds_store_2addr_b32 v59, v35, v36 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v37, v38 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v39, v40 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v33, 48, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v33
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB3_53
; %bb.52:
	v_mad_co_i64_i32 v[34:35], null, s12, v33, 0
	ds_load_b32 v38, v0
	v_lshlrev_b64_e32 v[36:37], 2, v[57:58]
	v_lshlrev_b64_e32 v[34:35], 2, v[34:35]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v34, s6, s16, v34
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, s17, v35, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v34, s6, v34, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v35, null, v35, v37, s6
	s_wait_dscnt 0x0
	global_store_b32 v[34:35], v38, off
.LBB3_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v34, 0x70, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v34
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_118
; %bb.54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_119
.LBB3_55:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_120
.LBB3_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_121
.LBB3_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_122
.LBB3_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_123
.LBB3_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_61
.LBB3_60:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:8960
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:384
.LBB3_61:                               ; %.preheader496.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v25, v26 offset1:20
	ds_store_2addr_b32 v59, v27, v28 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v29, v30 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v31, v32 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v25, 16, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v25
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_63
; %bb.62:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s8, s16, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:64
.LBB3_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_65
; %bb.64:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:1280
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s8, s16, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s8, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s8
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:64
.LBB3_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v25, 48, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v25
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_67
; %bb.66:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:2560
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s9, s16, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:192
.LBB3_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_69
; %bb.68:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:3840
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s9, s16, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s9, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s9
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:192
.LBB3_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v25, 0x50, v57
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v25
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_71
; %bb.70:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:5120
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v25, s10, s16, v25
	v_add_co_ci_u32_e64 v26, null, s17, v26, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:320
.LBB3_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_73
; %bb.72:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:6400
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s10, s16, v25
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, s10, v25, v27
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, s10
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:320
.LBB3_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v25, 0x70, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v25
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB3_75
; %bb.74:
	v_mad_co_i64_i32 v[25:26], null, s12, v60, 0
	ds_load_b32 v29, v0 offset:7680
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:448
.LBB3_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_77
; %bb.76:
	v_mad_co_i64_i32 v[25:26], null, s12, v61, 0
	ds_load_b32 v29, v0 offset:8960
	v_lshlrev_b64_e32 v[27:28], 2, v[57:58]
	v_lshlrev_b64_e32 v[25:26], 2, v[25:26]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, s16, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s17, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v25, vcc_lo, v25, v27
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, v26, v28, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[25:26], v29, off offset:448
.LBB3_77:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v17, v18 offset1:20
	ds_store_2addr_b32 v59, v19, v20 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v21, v22 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v23, v24 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_124
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_125
.LBB3_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_126
.LBB3_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_127
.LBB3_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_128
.LBB3_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_129
.LBB3_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_130
.LBB3_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_86
.LBB3_85:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:8960
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:448
.LBB3_86:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v9, v10 offset1:20
	ds_store_2addr_b32 v59, v11, v12 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v13, v14 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v15, v16 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_131
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_132
.LBB3_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_133
.LBB3_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_134
.LBB3_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_135
.LBB3_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_136
.LBB3_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_137
.LBB3_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_95
.LBB3_94:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:8960
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:448
.LBB3_95:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v59, v1, v2 offset1:20
	ds_store_2addr_b32 v59, v3, v4 offset0:40 offset1:60
	ds_store_2addr_b32 v59, v5, v6 offset0:80 offset1:100
	ds_store_2addr_b32 v59, v7, v8 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_138
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_139
.LBB3_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_140
.LBB3_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_141
.LBB3_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_142
.LBB3_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_143
.LBB3_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_144
.LBB3_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_104
.LBB3_103:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v4, v0 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[1:2]
	v_lshlrev_b64_e32 v[2:3], 2, v[57:58]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s16, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v1, v3, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v4, off offset:448
.LBB3_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_105:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB3_106:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:1280
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_35
.LBB3_107:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:2560
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_36
.LBB3_108:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:3840
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_37
.LBB3_109:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:5120
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_38
.LBB3_110:
	v_mad_co_i64_i32 v[51:52], null, s12, v50, 0
	ds_load_b32 v55, v0 offset:6400
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_39
.LBB3_111:
	v_mad_co_i64_i32 v[51:52], null, s12, v49, 0
	ds_load_b32 v55, v0 offset:7680
	v_lshlrev_b64_e32 v[53:54], 2, v[57:58]
	v_lshlrev_b64_e32 v[51:52], 2, v[51:52]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, s16, v51
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, s17, v52, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v51, s3, v51, v53
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v52, null, v52, v54, s3
	s_wait_dscnt 0x0
	global_store_b32 v[51:52], v55, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_40
	s_branch .LBB3_41
.LBB3_112:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:1280
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_45
.LBB3_113:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:2560
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_46
.LBB3_114:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:3840
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_47
.LBB3_115:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:5120
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_48
.LBB3_116:
	v_mad_co_i64_i32 v[43:44], null, s12, v42, 0
	ds_load_b32 v47, v0 offset:6400
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_49
.LBB3_117:
	v_mad_co_i64_i32 v[43:44], null, s12, v41, 0
	ds_load_b32 v47, v0 offset:7680
	v_lshlrev_b64_e32 v[45:46], 2, v[57:58]
	v_lshlrev_b64_e32 v[43:44], 2, v[43:44]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, s16, v43
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, s17, v44, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v43, s5, v43, v45
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, v44, v46, s5
	s_wait_dscnt 0x0
	global_store_b32 v[43:44], v47, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_50
	s_branch .LBB3_51
.LBB3_118:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:1280
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB3_55
.LBB3_119:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:2560
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_56
.LBB3_120:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:3840
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_57
.LBB3_121:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:5120
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_58
.LBB3_122:
	v_mad_co_i64_i32 v[35:36], null, s12, v34, 0
	ds_load_b32 v39, v0 offset:6400
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_59
.LBB3_123:
	v_mad_co_i64_i32 v[35:36], null, s12, v33, 0
	ds_load_b32 v39, v0 offset:7680
	v_lshlrev_b64_e32 v[37:38], 2, v[57:58]
	v_lshlrev_b64_e32 v[35:36], 2, v[35:36]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, s16, v35
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, s17, v36, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v35, s7, v35, v37
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v36, null, v36, v38, s7
	s_wait_dscnt 0x0
	global_store_b32 v[35:36], v39, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_60
	s_branch .LBB3_61
.LBB3_124:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_79
.LBB3_125:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:1280
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_80
.LBB3_126:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:2560
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_81
.LBB3_127:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:3840
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_82
.LBB3_128:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:5120
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_83
.LBB3_129:
	v_mad_co_i64_i32 v[17:18], null, s12, v50, 0
	ds_load_b32 v21, v0 offset:6400
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_84
.LBB3_130:
	v_mad_co_i64_i32 v[17:18], null, s12, v49, 0
	ds_load_b32 v21, v0 offset:7680
	v_lshlrev_b64_e32 v[19:20], 2, v[57:58]
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, s16, v17
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v17, vcc_lo, v17, v19
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, v18, v20, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[17:18], v21, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_85
	s_branch .LBB3_86
.LBB3_131:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_88
.LBB3_132:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:1280
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_89
.LBB3_133:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:2560
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_90
.LBB3_134:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:3840
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
.LBB3_135:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:5120
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_92
.LBB3_136:
	v_mad_co_i64_i32 v[9:10], null, s12, v42, 0
	ds_load_b32 v13, v0 offset:6400
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_93
.LBB3_137:
	v_mad_co_i64_i32 v[9:10], null, s12, v41, 0
	ds_load_b32 v13, v0 offset:7680
	v_lshlrev_b64_e32 v[11:12], 2, v[57:58]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v9, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_94
	s_branch .LBB3_95
.LBB3_138:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_97
.LBB3_139:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:1280
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_98
.LBB3_140:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:2560
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_99
.LBB3_141:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:3840
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_100
.LBB3_142:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:5120
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_101
.LBB3_143:
	v_mad_co_i64_i32 v[1:2], null, s12, v34, 0
	ds_load_b32 v5, v0 offset:6400
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_102
.LBB3_144:
	v_mad_co_i64_i32 v[1:2], null, s12, v33, 0
	ds_load_b32 v5, v0 offset:7680
	v_lshlrev_b64_e32 v[3:4], 2, v[57:58]
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v1, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v2, v4, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[1:2], v5, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_103
	s_branch .LBB3_104
.Lfunc_end3:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_set, .Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 68
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_set)<<4)&4080)>>4
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_vgpr, 256
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.private_seg_size, 68
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18820
; TotalNumSgprs: 40
; NumVgprs: 256
; ScratchSize: 68
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 256
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.type	__hip_cuid_18e58617ebcb2f7e,@object ; @__hip_cuid_18e58617ebcb2f7e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_18e58617ebcb2f7e
__hip_cuid_18e58617ebcb2f7e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_18e58617ebcb2f7e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_18e58617ebcb2f7e
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
    .private_segment_fixed_size: 72
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 38
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
    .private_segment_fixed_size: 68
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 18
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
    .private_segment_fixed_size: 68
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 18
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
