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
	s_lshl_b32 s22, ttmp7, 7
	v_lshrrev_b32_e32 v120, 5, v0
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
; %bb.2:                                ; %.preheader478.i
	s_ashr_i32 s0, s13, 31
	s_add_co_i32 s2, s12, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s10, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s22, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_5:                                ; %.preheader473.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s11, v1
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
.LBB1_6:                                ; %Flow1908
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v156, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s22, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s11, v9
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v144, 0 :: v_dual_and_b32 v13, 16, v13
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
	v_mov_b32_e32 v180, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_dual_mov_b32 v178, 0 :: v_dual_and_b32 v177, 15, v0
	v_mov_b32_e32 v154, 0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v141, 0
	v_bfe_u32 v168, v0, 4, 1
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v158, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v147, 0
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
	s_cbranch_scc1 .LBB1_17
; %bb.7:                                ; %.preheader472.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v185, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v2, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v5, s11, v1
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v10, s22, v1
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v7, 6, v168
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v177
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 8, v3
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_add3_u32 v190, 0, v1, v11
	v_lshl_add_u32 v1, v4, 11, 0
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v3, 10, v2
	v_min_i32_e32 v189, s4, v10
	v_cmp_gt_i32_e64 s2, s14, v10
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v191, 4, v6
	v_add3_u32 v192, 0, v12, v7
	v_add3_u32 v193, 0, v13, v8
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v194, 2, v6
	v_dual_mov_b32 v148, 0 :: v_dual_add_nc_u32 v195, v1, v2
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s29, 0x1000
	s_movk_i32 s25, 0x3000
	s_movk_i32 s26, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB1_9
.LBB1_8:                                ;   in Loop: Header=BB1_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s28
	s_mov_b32 s6, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_9:                                ; %.preheader472.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_11 Depth 2
	s_mov_b32 s7, s5
	s_add_co_i32 s27, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s27, s10
	s_mov_b32 s33, 0
	s_cselect_b32 s28, -1, 0
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s30, -1
	s_mov_b32 s31, 0
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=2
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v182, v182, v59 :: v_dual_add_f32 v179, v179, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v180, v180, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_dual_add_f32 v176, v176, v49 :: v_dual_mul_f32 v49, v57, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
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
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_cvt_f32_i32_e32 v41, v41
	v_dual_add_f32 v172, v172, v50 :: v_dual_add_f32 v169, v169, v52
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v170, v170, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_dual_mul_f32 v52, v109, v90 :: v_dual_mul_f32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v166, v166, v41
	v_cvt_f32_i32_e32 v45, v45
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	v_dual_add_f32 v167, v167, v42 :: v_dual_add_f32 v164, v164, v43
	s_delay_alu instid0(VALU_DEP_3)
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v165, v165, v41 :: v_dual_add_f32 v162, v162, v42
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
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_f32 v150, v150, v25 :: v_dual_add_f32 v151, v151, v26
	v_mul_f32_e32 v26, v94, v83
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v28, v29, v95
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v148, v148, v25 :: v_dual_mul_f32 v25, v26, v30
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	v_mul_f32_e32 v30, v94, v76
	v_dual_add_f32 v147, v147, v28 :: v_dual_mul_f32 v28, v94, v75
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
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
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v130, v130, v9
	s_barrier_signal -1
	v_mul_f32_e32 v9, v10, v11
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_xor_b32 s4, s30, -1
	s_mov_b32 s33, 1
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
	s_mov_b32 s30, 0
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
	s_mov_b32 s31, -1
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
	s_cbranch_vccnz .LBB1_8
.LBB1_11:                               ;   Parent Loop BB1_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s33, s7
	s_lshl_b32 s34, s33, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_mov_b32 s35, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[8:9], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	v_add_nc_u32_e32 v80, s29, v188
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s38, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s38
	v_add_co_u32 v3, s38, s34, v69
	v_add_co_u32 v5, s36, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s38
	v_add_co_u32 v7, s34, s34, v71
	v_add_co_ci_u32_e64 v6, null, s37, 0, s36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s35, 0, s34
	global_load_b64 v[116:117], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[118:119], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v195 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v195 offset0:2 offset1:3
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
	ds_load_2addr_b64 v[76:79], v195 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
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
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_and_b32 s29, s31, s28
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s29
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s0
	v_cndmask_b32_e64 v72, 0, v116, s0
	ds_store_2addr_stride64_b64 v186, v[72:73], v[112:113] offset1:16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s1
	v_cndmask_b32_e64 v72, 0, v118, s1
	ds_store_2addr_stride64_b64 v187, v[72:73], v[114:115] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=2
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
	v_mad_co_i64_i32 v[76:77], null, 0x48, v189, s[34:35]
	v_add_co_u32 v72, s38, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s38
	s_add_nc_u64 s[36:37], s[16:17], s[36:37]
	s_lshl_b32 s38, s33, 6
	s_mulk_i32 s4, 0x88
	v_add_co_u32 v76, vcc_lo, v76, v194
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[36:37], s[38:39]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s4
	v_add_co_u32 v74, s40, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s34, s36, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s35, 0, s40
	v_add_co_ci_u32_e64 v79, null, s37, 0, s34
	v_add_co_u32 v80, s34, s36, v71
	s_lshl_b32 s4, s33, 2
	v_add_co_ci_u32_e64 v81, null, s37, 0, s34
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[116:117], v[72:73], off offset:8
	global_load_b64 v[118:119], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB1_13:                               ; %.preheader470.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	v_add_nc_u32_e32 v84, 0, v188
	s_xor_b32 s4, s29, -1
	s_and_b32 s29, s30, exec_lo
	s_cselect_b32 s29, s25, 0x3400
	s_cselect_b32 s33, s26, 0x3c00
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v195 offset1:1
	ds_load_2addr_stride64_b64 v[80:83], v195 offset0:2 offset1:3
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
	v_add_nc_u32_e32 v76, 0x100, v84
	ds_load_2addr_b64 v[72:75], v195 offset0:32 offset1:96
	ds_load_2addr_stride64_b64 v[76:79], v76 offset0:16 offset1:17
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[74:75], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[78:79], v[72:73], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[78:79], v[74:75], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[78:79], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[78:79], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s33, v192
	v_add_nc_u32_e32 v72, s29, v193
	s_movk_i32 s29, 0x2000
	s_and_not1_b32 vcc_lo, exec_lo, s4
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
; %bb.14:                               ; %.preheader471.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	v_lshrrev_b32_e32 v198, v191, v197
	s_and_b32 s4, s31, exec_lo
	s_cselect_b32 s4, s25, 0x3400
	v_cndmask_b32_e64 v117, 0, v117, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s4, v190
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s4, s26, 0x3c00
	v_cndmask_b32_e64 v116, 0, v116, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v119, 0, v119, s1
	v_cndmask_b32_e64 v118, 0, v118, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s4, v190
	s_movk_i32 s29, 0x1000
	ds_store_2addr_stride64_b64 v186, v[116:117], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[118:119], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB1_10
.LBB1_15:
	s_mov_b32 s24, -1
.LBB1_16:                               ; %Flow1914
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s24
	s_cbranch_vccnz .LBB1_107
	s_branch .LBB1_211
.LBB1_17:                               ; %.preheader468.i
	v_mul_u32_u24_e32 v1, 0x500, v120
	v_lshlrev_b32_e32 v2, 2, v177
	v_mul_u32_u24_e32 v6, 0x50, v177
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add3_u32 v4, 0, v1, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_mad_u32_u24 v2, 0x280, v168, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_and_b32_e32 v3, 15, v1
	v_or_b32_e32 v1, s11, v177
	ds_store_2addr_b32 v2, v178, v185 offset1:20
	ds_store_2addr_b32 v2, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v5, s22, v3
	v_lshlrev_b32_e32 v3, 2, v3
	v_cmp_gt_i32_e64 s7, s12, v1
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_gt_i32_e32 vcc_lo, s14, v5
	v_add3_u32 v3, 0, v6, v3
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_19
; %bb.18:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s0, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s0
	v_add_co_u32 v6, s0, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s0
	ds_load_b32 v9, v3
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off
.LBB1_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 64, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v6
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_21
; %bb.20:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:1280
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off
.LBB1_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v7
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_23
; %bb.22:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:2560
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB1_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_25
; %bb.24:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:3840
	global_load_b32 v9, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:128
.LBB1_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v7
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_27
; %bb.26:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:5120
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB1_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_29
; %bb.28:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:6400
	global_load_b32 v9, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:256
.LBB1_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v7
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_31
; %bb.30:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v3 offset:7680
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB1_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v7, 0x280, v168
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_33
; %bb.32:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	v_add_co_u32 v8, s1, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	ds_load_b32 v11, v3 offset:8960
	global_load_b32 v10, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:384
.LBB1_33:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v4, v4, v7
	v_or_b32_e32 v7, 16, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v7
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v176, v175 offset1:20
	ds_store_2addr_b32 v4, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v172, v171 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v169, v170 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_35
; %bb.34:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s2, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s2
	v_add_co_u32 v8, s2, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	ds_load_b32 v11, v3
	global_load_b32 v10, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off
.LBB1_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 0x50, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v8
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_214
; %bb.36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_215
.LBB1_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_216
.LBB1_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_217
.LBB1_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_218
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_219
.LBB1_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_43
.LBB1_42:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:8960
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:384
.LBB1_43:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v9, 32, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v9
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v166, v167 offset1:20
	ds_store_2addr_b32 v4, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_45
; %bb.44:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s4, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s4
	v_add_co_u32 v10, s4, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s4
	ds_load_b32 v13, v3
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off
.LBB1_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v10, 0x60, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v10
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_220
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_221
.LBB1_47:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_222
.LBB1_48:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_223
.LBB1_49:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_224
.LBB1_50:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_225
.LBB1_51:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_53
.LBB1_52:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:8960
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384
.LBB1_53:                               ; %.preheader.3.i
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v11, 48, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v11
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v158, v159 offset1:20
	ds_store_2addr_b32 v4, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s15, s6
	s_cbranch_execz .LBB1_55
; %bb.54:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s6, s20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s6
	v_add_co_u32 v12, s6, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s6
	ds_load_b32 v15, v3
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off
.LBB1_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v12, 0x70, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v12
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_226
; %bb.56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_227
.LBB1_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_228
.LBB1_58:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_229
.LBB1_59:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_230
.LBB1_60:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_231
.LBB1_61:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_63
.LBB1_62:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:8960
	global_load_b32 v15, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:384
.LBB1_63:                               ; %.preheader467.1.i
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v13, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v13
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v150, v151 offset1:20
	ds_store_2addr_b32 v4, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v147, v146 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_65
; %bb.64:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s8, s20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s8
	v_add_co_u32 v13, s8, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s8
	ds_load_b32 v16, v3
	global_load_b32 v15, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:64
.LBB1_65:
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_67
; %bb.66:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s8, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s8
	v_add_co_u32 v13, s8, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s8
	ds_load_b32 v16, v3 offset:1280
	global_load_b32 v15, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:64
.LBB1_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v13, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v13
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_69
; %bb.68:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s9, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s9
	v_add_co_u32 v13, s9, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s9
	ds_load_b32 v16, v3 offset:2560
	global_load_b32 v15, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:192
.LBB1_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_71
; %bb.70:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s9, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s9
	v_add_co_u32 v13, s9, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s9
	ds_load_b32 v16, v3 offset:3840
	global_load_b32 v15, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:192
.LBB1_71:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v13, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v13
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_73
; %bb.72:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s10, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s10
	v_add_co_u32 v13, s10, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s10
	ds_load_b32 v16, v3 offset:5120
	global_load_b32 v15, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:320
.LBB1_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_75
; %bb.74:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s10, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s10
	v_add_co_u32 v13, s10, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s10
	ds_load_b32 v16, v3 offset:6400
	global_load_b32 v15, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:320
.LBB1_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v13, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s12, v13
	s_and_b32 s25, s10, vcc_lo
	s_and_saveexec_b32 s15, s25
	s_cbranch_execz .LBB1_77
; %bb.76:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v5, vcc_lo, s20, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, vcc_lo
	v_add_co_u32 v13, vcc_lo, v5, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, vcc_lo
	ds_load_b32 v15, v3 offset:7680
	global_load_b32 v5, v[13:14], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v15, v5
	global_store_b32 v[13:14], v5, off offset:448
.LBB1_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s15, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_79
; %bb.78:
	v_mad_co_i64_i32 v[5:6], null, s12, v6, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:8960
	global_load_b32 v13, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:448
.LBB1_79:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s15, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v142, v143 offset1:20
	ds_store_2addr_b32 v4, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_232
; %bb.80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_233
.LBB1_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_234
.LBB1_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_235
.LBB1_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_236
.LBB1_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_237
.LBB1_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_238
.LBB1_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_88
.LBB1_87:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:8960
	global_load_b32 v7, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:448
.LBB1_88:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v134, v135 offset1:20
	ds_store_2addr_b32 v4, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_239
; %bb.89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_240
.LBB1_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_241
.LBB1_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_242
.LBB1_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_243
.LBB1_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_244
.LBB1_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_245
.LBB1_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_97
.LBB1_96:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:8960
	global_load_b32 v7, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:448
.LBB1_97:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v126, v127 offset1:20
	ds_store_2addr_b32 v4, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_246
; %bb.98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_247
.LBB1_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_248
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_249
.LBB1_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_250
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_251
.LBB1_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_252
.LBB1_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_106
.LBB1_105:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	ds_load_b32 v3, v3 offset:8960
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v5, v2, vcc_lo
	global_load_b32 v4, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v3, v4
	global_store_b32 v[1:2], v3, off offset:448
.LBB1_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
	s_and_b32 vcc_lo, exec_lo, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_211
.LBB1_107:
	s_and_b32 vcc_lo, exec_lo, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_211
; %bb.108:                              ; %.preheader478.i18
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
	s_cbranch_execz .LBB1_112
; %bb.109:                              ; %.lr.ph.i300
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s22, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB1_111
; %bb.110:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_111:                              ; %.preheader473.loopexit.i301
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s11, v1
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
.LBB1_112:                              ; %Flow1913
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v156, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s22, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s11, v9
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v144, 0 :: v_dual_and_b32 v13, 16, v13
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
	v_mov_b32_e32 v180, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_dual_mov_b32 v178, 0 :: v_dual_and_b32 v177, 15, v0
	v_mov_b32_e32 v154, 0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v141, 0
	v_bfe_u32 v168, v0, 4, 1
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v158, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v184, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v147, 0
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
	s_cbranch_scc1 .LBB1_121
; %bb.113:                              ; %.preheader472.lr.ph.i191
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v185, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v2, 31, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v5, s11, v1
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v10, s22, v1
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v7, 6, v168
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v177
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 8, v3
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_add3_u32 v190, 0, v1, v11
	v_lshl_add_u32 v1, v4, 11, 0
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v3, 10, v2
	v_min_i32_e32 v189, s4, v10
	v_cmp_gt_i32_e64 s2, s14, v10
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v191, 4, v6
	v_add3_u32 v192, 0, v12, v7
	v_add3_u32 v193, 0, v13, v8
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v194, 2, v6
	v_dual_mov_b32 v148, 0 :: v_dual_add_nc_u32 v195, v1, v2
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB1_115
.LBB1_114:                              ;   in Loop: Header=BB1_115 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s6, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_121
.LBB1_115:                              ; %.preheader472.i196
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_117 Depth 2
	s_mov_b32 s7, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s24, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_lshl_b32 s7, s6, 1
	s_cmp_eq_u32 s24, s10
	s_mov_b32 s29, 0
	s_cselect_b32 s25, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_mov_b32 s27, -1
	s_mov_b32 s28, 0
	s_branch .LBB1_117
.LBB1_116:                              ;   in Loop: Header=BB1_117 Depth=2
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v182, v182, v59 :: v_dual_add_f32 v179, v179, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v180, v180, v62
	v_cvt_f32_i32_e32 v59, v93
	v_mul_f32_e32 v49, v57, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v111, v92 :: v_dual_mul_f32 v50, v58, v50
	v_mul_f32_e32 v58, v106, v92
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_fmac_f32_e32 v49, v57, v59
	v_dual_mul_f32 v57, v104, v92 :: v_dual_mul_f32 v60, v109, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v51, v58, v51 :: v_dual_mul_f32 v58, v107, v92
	v_dual_add_f32 v176, v176, v49 :: v_dual_mul_f32 v49, v57, v52
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v50, v60, v59
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
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_cvt_f32_i32_e32 v41, v41
	v_dual_add_f32 v172, v172, v50 :: v_dual_add_f32 v169, v169, v52
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v170, v170, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_dual_mul_f32 v52, v109, v90 :: v_dual_mul_f32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v166, v166, v41
	v_cvt_f32_i32_e32 v45, v45
	v_dual_mul_f32 v41, v49, v44 :: v_dual_fmac_f32 v42, v52, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v44, v105, v90
	v_cvt_f32_i32_e32 v46, v46
	v_dual_add_f32 v167, v167, v42 :: v_dual_add_f32 v164, v164, v43
	s_delay_alu instid0(VALU_DEP_3)
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v165, v165, v41 :: v_dual_add_f32 v162, v162, v42
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
	v_dual_mul_f32 v33, v94, v82 :: v_dual_mul_f32 v34, v94, v84
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_add_f32 v150, v150, v25 :: v_dual_add_f32 v151, v151, v26
	v_mul_f32_e32 v26, v94, v83
	v_mul_f32_e32 v25, v33, v27
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	v_dual_mul_f32 v28, v94, v80 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v17, v17
	v_fmac_f32_e32 v28, v29, v95
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_add_f32 v148, v148, v25 :: v_dual_mul_f32 v25, v26, v30
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	v_mul_f32_e32 v30, v94, v76
	v_dual_add_f32 v147, v147, v28 :: v_dual_mul_f32 v28, v94, v75
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
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
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v130, v130, v9
	s_barrier_signal -1
	v_mul_f32_e32 v9, v10, v11
	v_dual_mul_f32 v10, v90, v75 :: v_dual_mul_f32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	s_xor_b32 s4, s27, -1
	s_mov_b32 s29, 1
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
	s_mov_b32 s27, 0
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
	s_mov_b32 s28, -1
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
	s_cbranch_vccnz .LBB1_114
.LBB1_117:                              ;   Parent Loop BB1_115 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s4, s29, s7
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_mov_b32 s31, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[8:9], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	v_add_nc_u32_e32 v80, s26, v188
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s33
	v_add_co_u32 v3, s33, s30, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s31, 0, s33
	v_add_co_u32 v5, s33, s34, v70
	v_add_co_u32 v7, s30, s30, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s33
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	global_load_b64 v[116:117], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[118:119], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v195 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v195 offset0:2 offset1:3
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
	ds_load_2addr_b64 v[76:79], v195 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
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
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s0
	v_cndmask_b32_e64 v72, 0, v116, s0
	ds_store_2addr_stride64_b64 v186, v[72:73], v[112:113] offset1:16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s1
	v_cndmask_b32_e64 v72, 0, v118, s1
	ds_store_2addr_stride64_b64 v187, v[72:73], v[114:115] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_119
; %bb.118:                              ;   in Loop: Header=BB1_117 Depth=2
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
	v_mad_co_i64_i32 v[76:77], null, 0x48, v189, s[30:31]
	s_add_nc_u64 s[34:35], s[16:17], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s4, 0x88
	v_add_co_u32 v72, s33, s30, v68
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v73, null, s31, 0, s33
	v_add_co_u32 v76, vcc_lo, v76, v194
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s4
	v_add_co_u32 v74, s33, s30, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s30, s34, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	v_add_co_u32 v80, s30, s34, v71
	s_lshl_b32 s4, s29, 2
	v_add_co_ci_u32_e64 v81, null, s35, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[116:117], v[72:73], off offset:8
	global_load_b64 v[118:119], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB1_119:                              ; %.preheader470.i223
                                        ;   in Loop: Header=BB1_117 Depth=2
	v_add_nc_u32_e32 v84, 0, v188
	s_xor_b32 s4, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	s_cselect_b32 s29, s23, 0x3c00
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v195 offset1:1
	ds_load_2addr_stride64_b64 v[80:83], v195 offset0:2 offset1:3
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
	v_add_nc_u32_e32 v76, 0x100, v84
	ds_load_2addr_b64 v[72:75], v195 offset0:32 offset1:96
	ds_load_2addr_stride64_b64 v[76:79], v76 offset0:16 offset1:17
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[74:75], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[78:79], v[72:73], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[78:79], v[74:75], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[78:79], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[78:79], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s29, v192
	v_add_nc_u32_e32 v72, s26, v193
	s_movk_i32 s26, 0x2000
	s_and_not1_b32 vcc_lo, exec_lo, s4
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_116
; %bb.120:                              ; %.preheader471.i292
                                        ;   in Loop: Header=BB1_117 Depth=2
	v_lshrrev_b32_e32 v198, v191, v197
	s_and_b32 s4, s28, exec_lo
	s_cselect_b32 s4, s13, 0x3400
	v_cndmask_b32_e64 v117, 0, v117, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s4, v190
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s4, s23, 0x3c00
	v_cndmask_b32_e64 v116, 0, v116, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v119, 0, v119, s1
	v_cndmask_b32_e64 v118, 0, v118, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s4, v190
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[116:117], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[118:119], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB1_116
.LBB1_121:                              ; %.preheader468.i24
	v_mul_u32_u24_e32 v1, 0x500, v120
	v_lshlrev_b32_e32 v2, 2, v177
	v_lshrrev_b32_e32 v0, 4, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add3_u32 v3, 0, v1, v2
	v_and_b32_e32 v2, 15, v0
	v_or_b32_e32 v0, s11, v177
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_u32_u24 v1, 0x280, v168, v3
	v_or_b32_e32 v4, s22, v2
	v_lshlrev_b32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v0
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v177
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	s_delay_alu instid0(VALU_DEP_2)
	v_add3_u32 v2, 0, v1, v2
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_123
; %bb.122:
	v_mad_co_i64_i32 v[5:6], null, s12, v4, 0
	ds_load_b32 v9, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v1, s0, s20, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, s0
	v_add_co_u32 v5, s0, v1, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s0
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off
.LBB1_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_125
; %bb.124:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off
.LBB1_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v1
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_127
; %bb.126:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:128
.LBB1_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_129
; %bb.128:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:128
.LBB1_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v1
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_131
; %bb.130:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:256
.LBB1_131:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_133
; %bb.132:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:256
.LBB1_133:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v1
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_135
; %bb.134:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s1, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s1
	v_add_co_u32 v6, s1, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:384
.LBB1_135:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v168
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_137
; %bb.136:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB1_137:                              ; %.preheader.1.i62
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v6
	s_and_b32 s2, s7, s1
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_139
; %bb.138:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	ds_load_b32 v11, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s2, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s2
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off
.LBB1_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_253
; %bb.140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_254
.LBB1_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_255
.LBB1_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_256
.LBB1_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_257
.LBB1_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_258
.LBB1_145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_147
.LBB1_146:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:384
.LBB1_147:                              ; %.preheader.2.i73
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v8, 32, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v8
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v166, v167 offset1:20
	ds_store_2addr_b32 v3, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_149
; %bb.148:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s4, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s4
	v_add_co_u32 v9, s4, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s4
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB1_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_259
; %bb.150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_260
.LBB1_151:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_261
.LBB1_152:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_262
.LBB1_153:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_263
.LBB1_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_264
.LBB1_155:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_157
.LBB1_156:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB1_157:                              ; %.preheader.3.i84
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v10, 48, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v10
	s_and_b32 s6, s7, s5
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB1_159
; %bb.158:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s6, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s6
	v_add_co_u32 v11, s6, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s6
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB1_159:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_265
; %bb.160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_266
.LBB1_161:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_267
.LBB1_162:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_268
.LBB1_163:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_269
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_270
.LBB1_165:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_167
.LBB1_166:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
.LBB1_167:                              ; %.preheader467.1.i95
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v1, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v1
	s_and_b32 s8, s7, vcc_lo
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_169
; %bb.168:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s8, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s8
	v_add_co_u32 v12, s8, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB1_169:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_171
; %bb.170:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s8, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s8
	v_add_co_u32 v12, s8, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB1_171:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v1
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_173
; %bb.172:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s9, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s9
	v_add_co_u32 v12, s9, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB1_173:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_175
; %bb.174:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s9, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s9
	v_add_co_u32 v12, s9, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB1_175:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v1
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_177
; %bb.176:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s10, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s10
	v_add_co_u32 v12, s10, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB1_177:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_179
; %bb.178:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s10, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s10
	v_add_co_u32 v12, s10, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB1_179:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v1
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB1_181
; %bb.180:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v4, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, vcc_lo, s20, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, vcc_lo
	v_add_co_u32 v12, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v4, off offset:448
.LBB1_181:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_183
; %bb.182:
	v_mad_co_i64_i32 v[4:5], null, s12, v5, 0
	ds_load_b32 v14, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:448
.LBB1_183:                              ; %.preheader.1.1.i108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_271
; %bb.184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_272
.LBB1_185:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_273
.LBB1_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_274
.LBB1_187:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_275
.LBB1_188:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_276
.LBB1_189:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_277
.LBB1_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_192
.LBB1_191:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:448
.LBB1_192:                              ; %.preheader.2.1.i117
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_278
; %bb.193:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_279
.LBB1_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_280
.LBB1_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_281
.LBB1_196:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_282
.LBB1_197:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_283
.LBB1_198:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_284
.LBB1_199:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_201
.LBB1_200:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v8, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v8, off offset:448
.LBB1_201:                              ; %.preheader.3.1.i126
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
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
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_285
; %bb.202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_286
.LBB1_203:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_287
.LBB1_204:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_288
.LBB1_205:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_289
.LBB1_206:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_290
.LBB1_207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_291
.LBB1_208:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_210
.LBB1_209:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v5, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]
	v_add_co_u32 v2, vcc_lo, s20, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_add_co_u32 v0, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v5, off offset:448
.LBB1_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
.LBB1_211:                              ; %Flow1916
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_213
; %bb.212:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_213:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB1_214:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:1280
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_37
.LBB1_215:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:2560
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_38
.LBB1_216:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:3840
	global_load_b32 v11, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_39
.LBB1_217:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:5120
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_40
.LBB1_218:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:6400
	global_load_b32 v11, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_41
.LBB1_219:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v12, v3 offset:7680
	global_load_b32 v11, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_42
	s_branch .LBB1_43
.LBB1_220:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:1280
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_47
.LBB1_221:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:2560
	global_load_b32 v13, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_48
.LBB1_222:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:3840
	global_load_b32 v13, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_49
.LBB1_223:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:5120
	global_load_b32 v13, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_50
.LBB1_224:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:6400
	global_load_b32 v13, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_51
.LBB1_225:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v14, v3 offset:7680
	global_load_b32 v13, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off offset:384
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_52
	s_branch .LBB1_53
.LBB1_226:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:1280
	global_load_b32 v15, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execz .LBB1_57
.LBB1_227:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:2560
	global_load_b32 v15, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_58
.LBB1_228:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:3840
	global_load_b32 v15, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:128
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_59
.LBB1_229:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:5120
	global_load_b32 v15, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_60
.LBB1_230:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:6400
	global_load_b32 v15, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_61
.LBB1_231:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v13, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v16, v3 offset:7680
	global_load_b32 v15, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v16, v15
	global_store_b32 v[13:14], v15, off offset:384
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_62
	s_branch .LBB1_63
.LBB1_232:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3
	global_load_b32 v13, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_81
.LBB1_233:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:1280
	global_load_b32 v13, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_82
.LBB1_234:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:2560
	global_load_b32 v13, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_83
.LBB1_235:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:3840
	global_load_b32 v13, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_84
.LBB1_236:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:5120
	global_load_b32 v13, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_85
.LBB1_237:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v14, v3 offset:6400
	global_load_b32 v13, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[5:6], v13, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_86
.LBB1_238:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:7680
	global_load_b32 v7, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v13, v7
	global_store_b32 v[5:6], v7, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_87
	s_branch .LBB1_88
.LBB1_239:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3
	global_load_b32 v7, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_90
.LBB1_240:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:1280
	global_load_b32 v7, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_91
.LBB1_241:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:2560
	global_load_b32 v7, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_92
.LBB1_242:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:3840
	global_load_b32 v7, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_93
.LBB1_243:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:5120
	global_load_b32 v7, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_94
.LBB1_244:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:6400
	global_load_b32 v7, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_95
.LBB1_245:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v3 offset:7680
	global_load_b32 v7, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_96
	s_branch .LBB1_97
.LBB1_246:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3
	global_load_b32 v6, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_99
.LBB1_247:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:1280
	global_load_b32 v6, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_100
.LBB1_248:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:2560
	global_load_b32 v6, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_101
.LBB1_249:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:3840
	global_load_b32 v6, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_102
.LBB1_250:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:5120
	global_load_b32 v6, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_103
.LBB1_251:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:6400
	global_load_b32 v6, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_104
.LBB1_252:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v3 offset:7680
	global_load_b32 v6, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_105
	s_branch .LBB1_106
.LBB1_253:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_141
.LBB1_254:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_142
.LBB1_255:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_143
.LBB1_256:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_144
.LBB1_257:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_145
.LBB1_258:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s3, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s3
	v_add_co_u32 v8, s3, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_146
	s_branch .LBB1_147
.LBB1_259:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_151
.LBB1_260:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_152
.LBB1_261:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_153
.LBB1_262:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_154
.LBB1_263:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_155
.LBB1_264:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s5, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s5
	v_add_co_u32 v10, s5, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_156
	s_branch .LBB1_157
.LBB1_265:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB1_161
.LBB1_266:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_162
.LBB1_267:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_163
.LBB1_268:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_164
.LBB1_269:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_165
.LBB1_270:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s7, s20, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s7
	v_add_co_u32 v12, s7, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_166
	s_branch .LBB1_167
.LBB1_271:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_185
.LBB1_272:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_186
.LBB1_273:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_187
.LBB1_274:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_188
.LBB1_275:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_189
.LBB1_276:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_190
.LBB1_277:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v6, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v6, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_191
	s_branch .LBB1_192
.LBB1_278:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_194
.LBB1_279:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_195
.LBB1_280:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_196
.LBB1_281:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_197
.LBB1_282:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_198
.LBB1_283:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_199
.LBB1_284:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v8, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v1, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v1, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v8, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_200
	s_branch .LBB1_201
.LBB1_285:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_203
.LBB1_286:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_204
.LBB1_287:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_205
.LBB1_288:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_206
.LBB1_289:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_207
.LBB1_290:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_208
.LBB1_291:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v1, vcc_lo, s20, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s21, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_209
	s_branch .LBB1_210
.Lfunc_end1:
	.size	gemm_mq4g256v2_residual_mmq_iu4, .Lfunc_end1-gemm_mq4g256v2_residual_mmq_iu4
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
; codeLenInByte = 29212
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
	s_cbranch_vccnz .LBB2_104
; %bb.1:                                ; %.preheader478.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s2, s12, -1
	s_ashr_i32 s22, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s21, v2
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v3, 2, v1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB2_4:                                ; %.preheader473.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s20, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[4:5]
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
.LBB2_5:                                ; %Flow801
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s21, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s20, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s2, v5
	v_dual_mov_b32 v170, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s3, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s3, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v177, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v144, 0
	v_or_b32_e32 v15, 8, v177
	v_and_or_b32 v16, v177, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v150, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v176, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v171, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v167, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v154, 0
	v_add_nc_u32_e32 v186, 0, v15
	v_add_nc_u32_e32 v187, 0, v11
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
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
	s_cbranch_scc1 .LBB2_14
; %bb.6:                                ; %.preheader472.lr.ph.i
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v3, s20, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s2, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v185, 0 :: v_dual_add_nc_u32 v8, s21, v2
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_mad_co_u64_u32 v[65:66], null, s3, v7, s[4:5]
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 2, v1
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_ashrrev_i32_e32 v7, 31, v7
	v_min_i32_e32 v188, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v189, 0, v2, v9
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v2, 8, v5
	v_lshl_or_b32 v190, v5, 10, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v7, v[66:67]
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v5, 6, v167
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v7, 9, v6
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v8, 3, v176
	v_cmp_gt_i32_e64 s3, s12, v3
	v_lshl_add_u32 v3, v6, 11, 0
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v191, 4, v1
	v_add3_u32 v192, 0, v2, v5
	v_add3_u32 v193, 0, v7, v8
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v194, 2, v1
	v_dual_mov_b32 v150, 0 :: v_dual_add_nc_u32 v195, v3, v4
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	s_branch .LBB2_8
.LBB2_7:                                ;   in Loop: Header=BB2_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_8:                                ; %.preheader472.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s29, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_mov_b32 s27, -1
	s_mov_b32 s28, 0
	s_branch .LBB2_10
.LBB2_9:                                ;   in Loop: Header=BB2_10 Depth=2
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v182, v182, v59 :: v_dual_add_f32 v179, v179, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v180, v180, v62
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
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, v102, v92 :: v_dual_fmac_f32 v49, v52, v59
	v_add_f32_e32 v172, v172, v51
	v_mul_f32_e32 v50, v57, v53
	v_dual_mul_f32 v52, v96, v92 :: v_dual_mul_f32 v57, v98, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v55, v103, v92 :: v_dual_mul_f32 v54, v101, v92
	v_cvt_f32_i32_e32 v56, v56
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v51, v55, v59 :: v_dual_fmac_f32 v50, v54, v59
	v_mul_f32_e32 v55, v99, v92
	v_mul_f32_e32 v54, v57, v56
	v_fmac_f32_e32 v52, v53, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v173, v173, v49 :: v_dual_add_f32 v170, v170, v51
	v_add_f32_e32 v171, v171, v50
	v_dual_mul_f32 v49, v110, v90 :: v_dual_mul_f32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_dual_fmac_f32 v54, v55, v59 :: v_dual_add_f32 v169, v169, v52
	v_cvt_f32_i32_e32 v51, v91
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v168, v168, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_dual_mul_f32 v52, v109, v90 :: v_dual_mul_f32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_dual_add_f32 v165, v165, v41 :: v_dual_fmac_f32 v42, v52, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v41, v49, v44
	v_dual_mul_f32 v44, v105, v90 :: v_dual_fmac_f32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v166, v166, v42 :: v_dual_fmac_f32 v41, v44, v51
	v_dual_add_f32 v163, v163, v43 :: v_dual_mul_f32 v42, v49, v45
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
	v_dual_add_f32 v164, v164, v41 :: v_dual_add_f32 v161, v161, v42
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v41, v110, v72 :: v_dual_fmac_f32 v46, v47, v51
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_dual_add_f32 v162, v162, v43 :: v_dual_add_f32 v159, v159, v44
	v_add_f32_e32 v160, v160, v46
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v157, v157, v33
	v_dual_mul_f32 v33, v41, v36 :: v_dual_fmac_f32 v34, v44, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v36, v105, v72 :: v_dual_fmac_f32 v35, v42, v43
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v42, v102, v72
	v_cvt_f32_i32_e32 v37, v37
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
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_mul_f32 v38, v99, v72 :: v_dual_add_f32 v153, v153, v34
	v_fmac_f32_e32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v154, v154, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v151, v151, v33
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v149, v149, v25 :: v_dual_add_f32 v150, v150, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v147, v147, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v148, v148, v27
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	v_mul_f32_e32 v30, v94, v76
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_f32_e32 v131, v131, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	v_add_f32_e32 v152, v152, v36
	s_xor_b32 s8, s27, -1
	s_mov_b32 s29, 1
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	v_mul_f32_e32 v10, v90, v75
	s_mov_b32 s28, -1
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
	s_cbranch_vccnz .LBB2_7
.LBB2_10:                               ;   Parent Loop BB2_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s29, s11
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_mov_b32 s31, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	v_add_nc_u32_e32 v80, s26, v190
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s33
	v_add_co_u32 v3, s33, s30, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s31, 0, s33
	v_add_co_u32 v5, s33, s34, v70
	v_add_co_u32 v7, s30, s30, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s33
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	global_load_b64 v[116:117], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[118:119], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v195 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v195 offset0:2 offset1:3
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
	ds_load_2addr_b64 v[76:79], v195 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
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
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s0
	v_cndmask_b32_e64 v72, 0, v116, s0
	ds_store_2addr_stride64_b64 v186, v[72:73], v[112:113] offset1:16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s1
	v_cndmask_b32_e64 v72, 0, v118, s1
	ds_store_2addr_stride64_b64 v187, v[72:73], v[114:115] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
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
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[30:31]
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s8, 0x88
	v_add_co_u32 v72, s33, s30, v68
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v73, null, s31, 0, s33
	v_add_co_u32 v76, vcc_lo, v76, v194
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_u32 v74, s33, s30, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s30, s34, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	v_add_co_u32 v80, s30, s34, v71
	s_lshl_b32 s8, s29, 2
	v_add_co_ci_u32_e64 v81, null, s35, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[116:117], v[72:73], off offset:8
	global_load_b64 v[118:119], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB2_12:                               ; %.preheader470.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	s_cselect_b32 s29, s23, 0x3c00
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v195 offset1:1
	ds_load_2addr_stride64_b64 v[80:83], v195 offset0:2 offset1:3
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
	v_add_nc_u32_e32 v76, 0x100, v84
	ds_load_2addr_b64 v[72:75], v195 offset0:32 offset1:96
	ds_load_2addr_stride64_b64 v[76:79], v76 offset0:16 offset1:17
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[74:75], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[78:79], v[72:73], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[78:79], v[74:75], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[78:79], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[78:79], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s29, v192
	v_add_nc_u32_e32 v72, s26, v193
	s_movk_i32 s26, 0x2000
	s_and_not1_b32 vcc_lo, exec_lo, s8
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_9
; %bb.13:                               ; %.preheader471.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	v_lshrrev_b32_e32 v198, v191, v197
	s_and_b32 s8, s28, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v117, 0, v117, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v189
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s8, s23, 0x3c00
	v_cndmask_b32_e64 v116, 0, v116, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v119, 0, v119, s1
	v_cndmask_b32_e64 v118, 0, v118, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s8, v189
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[116:117], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[118:119], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB2_9
.LBB2_14:                               ; %.preheader468.i
	v_mul_u32_u24_e32 v1, 0x500, v177
	v_lshlrev_b32_e32 v2, 2, v176
	v_lshrrev_b32_e32 v0, 4, v0
	v_mul_u32_u24_e32 v5, 0x50, v176
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add3_u32 v3, 0, v1, v2
	v_and_b32_e32 v2, 15, v0
	v_or_b32_e32 v0, s20, v176
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_u32_u24 v1, 0x280, v167, v3
	v_or_b32_e32 v4, s21, v2
	v_lshlrev_b32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v0
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	v_ashrrev_i32_e32 v1, 31, v0
	v_add3_u32 v2, 0, v5, v2
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_16
; %bb.15:
	v_mad_co_i64_i32 v[5:6], null, s12, v4, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s0, s16, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s17, v6, s0
	v_add_co_u32 v5, s0, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s0
	ds_load_b32 v8, v2
	global_load_b32 v7, v[5:6], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off
.LBB2_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_18
; %bb.17:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:1280
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off
.LBB2_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v6
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:2560
	global_load_b32 v8, v[6:7], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:128
.LBB2_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_22
; %bb.21:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:3840
	global_load_b32 v8, v[6:7], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:128
.LBB2_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v6
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_24
; %bb.23:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:5120
	global_load_b32 v8, v[6:7], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:256
.LBB2_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_26
; %bb.25:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:6400
	global_load_b32 v8, v[6:7], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:256
.LBB2_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v6
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_28
; %bb.27:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	ds_load_b32 v9, v2 offset:7680
	global_load_b32 v8, v[6:7], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:384
.LBB2_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v167
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_30
; %bb.29:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s1, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s1
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v10, v2 offset:8960
	global_load_b32 v9, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off offset:384
.LBB2_30:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v6
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v175, v174 offset1:20
	ds_store_2addr_b32 v3, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_32
; %bb.31:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, s2, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s2
	v_add_co_u32 v7, s2, v7, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	ds_load_b32 v10, v2
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v10, v9
	global_store_b32 v[7:8], v9, off
.LBB2_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_105
; %bb.33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_106
.LBB2_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_107
.LBB2_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_108
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_109
.LBB2_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_110
.LBB2_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_40
.LBB2_39:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:8960
	global_load_b32 v10, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:384
.LBB2_40:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v8, 32, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v8
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v165, v166 offset1:20
	ds_store_2addr_b32 v3, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v159, v160 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_42
; %bb.41:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, s4, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s4
	v_add_co_u32 v9, s4, v9, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s4
	ds_load_b32 v12, v2
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v12, v11
	global_store_b32 v[9:10], v11, off
.LBB2_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_111
; %bb.43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_112
.LBB2_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_113
.LBB2_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_114
.LBB2_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_115
.LBB2_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_116
.LBB2_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_50
.LBB2_49:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:8960
	global_load_b32 v12, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:384
.LBB2_50:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v10, 48, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v10
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v157, v158 offset1:20
	ds_store_2addr_b32 v3, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB2_52
; %bb.51:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, s6, s16, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s6
	v_add_co_u32 v11, s6, v11, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s6
	ds_load_b32 v14, v2
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v14, v13
	global_store_b32 v[11:12], v13, off
.LBB2_52:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_117
; %bb.53:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_118
.LBB2_54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_119
.LBB2_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_120
.LBB2_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_121
.LBB2_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_122
.LBB2_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_60
.LBB2_59:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:8960
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
.LBB2_60:                               ; %.preheader467.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v12
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v149, v150 offset1:20
	ds_store_2addr_b32 v3, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_62
; %bb.61:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s8, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s8
	v_add_co_u32 v12, s8, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	ds_load_b32 v15, v2
	global_load_b32 v14, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:64
.LBB2_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_64
; %bb.63:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s8, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s8
	v_add_co_u32 v12, s8, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	ds_load_b32 v15, v2 offset:1280
	global_load_b32 v14, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:64
.LBB2_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v12, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v12
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_66
; %bb.65:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s9, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s9
	v_add_co_u32 v12, s9, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	ds_load_b32 v15, v2 offset:2560
	global_load_b32 v14, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:192
.LBB2_66:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_68
; %bb.67:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s9, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s9
	v_add_co_u32 v12, s9, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	ds_load_b32 v15, v2 offset:3840
	global_load_b32 v14, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:192
.LBB2_68:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v12, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v12
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_70
; %bb.69:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s10, s16, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s10
	v_add_co_u32 v12, s10, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	ds_load_b32 v15, v2 offset:5120
	global_load_b32 v14, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:320
.LBB2_70:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_72
; %bb.71:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s10, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s10
	v_add_co_u32 v12, s10, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	ds_load_b32 v15, v2 offset:6400
	global_load_b32 v14, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:320
.LBB2_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v12, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v12
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB2_74
; %bb.73:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v4, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo
	v_add_co_u32 v12, vcc_lo, v4, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	ds_load_b32 v14, v2 offset:7680
	global_load_b32 v4, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v14, v4
	global_store_b32 v[12:13], v4, off offset:448
.LBB2_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_76
; %bb.75:
	v_mad_co_i64_i32 v[4:5], null, s12, v5, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:8960
	global_load_b32 v12, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:448
.LBB2_76:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v141, v142 offset1:20
	ds_store_2addr_b32 v3, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_123
; %bb.77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_124
.LBB2_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_125
.LBB2_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_126
.LBB2_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_127
.LBB2_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_128
.LBB2_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_129
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_85
.LBB2_84:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:8960
	global_load_b32 v6, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:448
.LBB2_85:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v133, v134 offset1:20
	ds_store_2addr_b32 v3, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_130
; %bb.86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_131
.LBB2_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_132
.LBB2_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_133
.LBB2_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_134
.LBB2_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_135
.LBB2_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_136
.LBB2_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_94
.LBB2_93:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:8960
	global_load_b32 v6, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:448
.LBB2_94:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v125, v126 offset1:20
	ds_store_2addr_b32 v3, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_137
; %bb.95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_138
.LBB2_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_139
.LBB2_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_140
.LBB2_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_141
.LBB2_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_142
.LBB2_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_143
.LBB2_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_103
.LBB2_102:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	ds_load_b32 v2, v2 offset:8960
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v3, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v4, v1, vcc_lo
	global_load_b32 v3, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v2, v3
	global_store_b32 v[0:1], v2, off offset:448
.LBB2_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_104:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB2_105:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:1280
	global_load_b32 v10, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_34
.LBB2_106:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:2560
	global_load_b32 v10, v[8:9], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_35
.LBB2_107:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:3840
	global_load_b32 v10, v[8:9], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_36
.LBB2_108:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:5120
	global_load_b32 v10, v[8:9], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_37
.LBB2_109:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:6400
	global_load_b32 v10, v[8:9], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_38
.LBB2_110:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	ds_load_b32 v11, v2 offset:7680
	global_load_b32 v10, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v11, v10
	global_store_b32 v[8:9], v10, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_39
	s_branch .LBB2_40
.LBB2_111:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:1280
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_44
.LBB2_112:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:2560
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_45
.LBB2_113:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:3840
	global_load_b32 v12, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_46
.LBB2_114:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:5120
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_47
.LBB2_115:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:6400
	global_load_b32 v12, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_48
.LBB2_116:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	ds_load_b32 v13, v2 offset:7680
	global_load_b32 v12, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[10:11], v12, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_49
	s_branch .LBB2_50
.LBB2_117:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:1280
	global_load_b32 v14, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB2_54
.LBB2_118:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:2560
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_55
.LBB2_119:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:3840
	global_load_b32 v14, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_56
.LBB2_120:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:5120
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_57
.LBB2_121:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:6400
	global_load_b32 v14, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_58
.LBB2_122:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	ds_load_b32 v15, v2 offset:7680
	global_load_b32 v14, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v15, v14
	global_store_b32 v[12:13], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_59
	s_branch .LBB2_60
.LBB2_123:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2
	global_load_b32 v12, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_78
.LBB2_124:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:1280
	global_load_b32 v12, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_79
.LBB2_125:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:2560
	global_load_b32 v12, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_80
.LBB2_126:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:3840
	global_load_b32 v12, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_81
.LBB2_127:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:5120
	global_load_b32 v12, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_82
.LBB2_128:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v13, v2 offset:6400
	global_load_b32 v12, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v13, v12
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_83
.LBB2_129:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	ds_load_b32 v12, v2 offset:7680
	global_load_b32 v6, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v12, v6
	global_store_b32 v[4:5], v6, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_84
	s_branch .LBB2_85
.LBB2_130:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2
	global_load_b32 v6, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_87
.LBB2_131:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:1280
	global_load_b32 v6, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_88
.LBB2_132:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:2560
	global_load_b32 v6, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_89
.LBB2_133:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:3840
	global_load_b32 v6, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_90
.LBB2_134:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:5120
	global_load_b32 v6, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_91
.LBB2_135:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:6400
	global_load_b32 v6, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_92
.LBB2_136:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v7, v2 offset:7680
	global_load_b32 v6, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v7, v6
	global_store_b32 v[4:5], v6, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_93
	s_branch .LBB2_94
.LBB2_137:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2
	global_load_b32 v5, v[3:4], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_96
.LBB2_138:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:1280
	global_load_b32 v5, v[3:4], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_97
.LBB2_139:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:2560
	global_load_b32 v5, v[3:4], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_98
.LBB2_140:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:3840
	global_load_b32 v5, v[3:4], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_99
.LBB2_141:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:5120
	global_load_b32 v5, v[3:4], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_100
.LBB2_142:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:6400
	global_load_b32 v5, v[3:4], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_101
.LBB2_143:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	ds_load_b32 v6, v2 offset:7680
	global_load_b32 v5, v[3:4], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v6, v5
	global_store_b32 v[3:4], v5, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_102
	s_branch .LBB2_103
.Lfunc_end2:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_add, .Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4_full_add
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
; codeLenInByte = 15008
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
	s_cbranch_vccnz .LBB3_104
; %bb.1:                                ; %.preheader478.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_ashr_i32 s0, s13, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s13, s0
	s_add_co_i32 s2, s12, -1
	s_ashr_i32 s22, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s21, v2
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
.LBB3_4:                                ; %.preheader473.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s20, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[4:5]
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
.LBB3_5:                                ; %Flow801
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s21, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s20, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s2, v5
	v_dual_mov_b32 v170, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s3, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s3, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v177, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v144, 0
	v_or_b32_e32 v15, 8, v177
	v_and_or_b32 v16, v177, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v150, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v176, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v171, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v167, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v154, 0
	v_add_nc_u32_e32 v186, 0, v15
	v_add_nc_u32_e32 v187, 0, v11
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
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
; %bb.6:                                ; %.preheader472.lr.ph.i
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v3, s20, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s2, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v185, 0 :: v_dual_add_nc_u32 v8, s21, v2
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_mad_co_u64_u32 v[65:66], null, s3, v7, s[4:5]
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 2, v1
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_ashrrev_i32_e32 v7, 31, v7
	v_min_i32_e32 v188, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v189, 0, v2, v9
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v2, 8, v5
	v_lshl_or_b32 v190, v5, 10, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v7, v[66:67]
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v5, 6, v167
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v7, 9, v6
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v8, 3, v176
	v_cmp_gt_i32_e64 s3, s12, v3
	v_lshl_add_u32 v3, v6, 11, 0
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v191, 4, v1
	v_add3_u32 v192, 0, v2, v5
	v_add3_u32 v193, 0, v7, v8
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v194, 2, v1
	v_dual_mov_b32 v150, 0 :: v_dual_add_nc_u32 v195, v3, v4
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_8:                                ; %.preheader472.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_10 Depth 2
	s_mov_b32 s11, s9
	s_add_co_i32 s24, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_lshl_b32 s11, s10, 1
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s29, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_mov_b32 s27, -1
	s_mov_b32 s28, 0
	s_branch .LBB3_10
.LBB3_9:                                ;   in Loop: Header=BB3_10 Depth=2
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v57, v110, v92 :: v_dual_fmac_f32 v62, v63, v95
	v_cvt_f32_i32_e32 v49, v49
	v_dual_add_f32 v182, v182, v59 :: v_dual_add_f32 v179, v179, v60
	v_mul_f32_e32 v58, v108, v92
	v_cvt_f32_i32_e32 v50, v50
	v_add_f32_e32 v180, v180, v62
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
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, v102, v92 :: v_dual_fmac_f32 v49, v52, v59
	v_add_f32_e32 v172, v172, v51
	v_mul_f32_e32 v50, v57, v53
	v_dual_mul_f32 v52, v96, v92 :: v_dual_mul_f32 v57, v98, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v51, v58, v54
	v_cvt_f32_i32_e32 v53, v55
	v_dual_mul_f32 v55, v103, v92 :: v_dual_mul_f32 v54, v101, v92
	v_cvt_f32_i32_e32 v56, v56
	v_cvt_f32_i32_e32 v41, v41
	v_dual_mul_f32 v52, v52, v53 :: v_dual_mul_f32 v53, v97, v92
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v51, v55, v59 :: v_dual_fmac_f32 v50, v54, v59
	v_mul_f32_e32 v55, v99, v92
	v_mul_f32_e32 v54, v57, v56
	v_fmac_f32_e32 v52, v53, v59
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v173, v173, v49 :: v_dual_add_f32 v170, v170, v51
	v_add_f32_e32 v171, v171, v50
	v_dual_mul_f32 v49, v110, v90 :: v_dual_mul_f32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_dual_fmac_f32 v54, v55, v59 :: v_dual_add_f32 v169, v169, v52
	v_cvt_f32_i32_e32 v51, v91
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v168, v168, v54 :: v_dual_fmac_f32 v41, v49, v51
	v_dual_mul_f32 v52, v109, v90 :: v_dual_mul_f32 v49, v104, v90
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v43, v50, v43 :: v_dual_mul_f32 v50, v107, v90
	v_dual_add_f32 v165, v165, v41 :: v_dual_fmac_f32 v42, v52, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v41, v49, v44
	v_dual_mul_f32 v44, v105, v90 :: v_dual_fmac_f32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v166, v166, v42 :: v_dual_fmac_f32 v41, v44, v51
	v_dual_add_f32 v163, v163, v43 :: v_dual_mul_f32 v42, v49, v45
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
	v_dual_add_f32 v164, v164, v41 :: v_dual_add_f32 v161, v161, v42
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v41, v110, v72 :: v_dual_fmac_f32 v46, v47, v51
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v42, v108, v72
	v_cvt_f32_i32_e32 v34, v34
	v_dual_add_f32 v162, v162, v43 :: v_dual_add_f32 v159, v159, v44
	v_add_f32_e32 v160, v160, v46
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
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v157, v157, v33
	v_dual_mul_f32 v33, v41, v36 :: v_dual_fmac_f32 v34, v44, v43
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v36, v105, v72 :: v_dual_fmac_f32 v35, v42, v43
	v_dual_mul_f32 v41, v100, v72 :: v_dual_mul_f32 v42, v102, v72
	v_cvt_f32_i32_e32 v37, v37
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
	v_dual_mul_f32 v39, v94, v88 :: v_dual_mul_f32 v40, v94, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_dual_mul_f32 v38, v99, v72 :: v_dual_add_f32 v153, v153, v34
	v_fmac_f32_e32 v33, v37, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v25, v39, v25 :: v_dual_add_f32 v154, v154, v35
	v_mul_f32_e32 v26, v40, v26
	v_dual_mul_f32 v34, v94, v89 :: v_dual_mul_f32 v37, v94, v87
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v36, v38, v43 :: v_dual_add_f32 v151, v151, v33
	v_mul_f32_e32 v33, v94, v82
	v_dual_fmac_f32 v25, v34, v95 :: v_dual_fmac_f32 v26, v37, v95
	v_cvt_f32_i32_e32 v27, v27
	v_mul_f32_e32 v34, v94, v84
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_add_f32 v149, v149, v25 :: v_dual_add_f32 v150, v150, v26
	v_mul_f32_e32 v25, v33, v27
	v_dual_mul_f32 v26, v94, v83 :: v_dual_mul_f32 v33, v94, v85
	v_cvt_f32_i32_e32 v29, v29
	v_mul_f32_e32 v27, v34, v28
	v_mul_f32_e32 v28, v94, v80
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_mul_f32_e32 v28, v28, v29
	v_mul_f32_e32 v29, v94, v81
	v_fmac_f32_e32 v25, v26, v95
	v_dual_mul_f32 v26, v94, v78 :: v_dual_fmac_f32 v27, v33, v95
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v28, v29, v95 :: v_dual_add_f32 v147, v147, v25
	v_dual_mul_f32 v25, v26, v30 :: v_dual_add_f32 v148, v148, v27
	v_dual_mul_f32 v26, v94, v79 :: v_dual_mul_f32 v29, v94, v74
	v_mul_f32_e32 v30, v94, v76
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_f32_e32 v131, v131, v11
	v_fmac_f32_e32 v13, v17, v51
	v_cvt_f32_i32_e32 v11, v15
	v_add_f32_e32 v152, v152, v36
	s_xor_b32 s8, s27, -1
	s_mov_b32 s29, 1
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	v_add_f32_e32 v129, v129, v9
	v_mul_f32_e32 v9, v10, v11
	v_mul_f32_e32 v11, v12, v14
	v_mul_f32_e32 v12, v72, v88
	v_mul_f32_e32 v10, v90, v75
	s_mov_b32 s28, -1
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
	s_or_b32 s8, s29, s11
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_mov_b32 s31, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	v_add_nc_u32_e32 v80, s26, v190
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s33, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s33
	v_add_co_u32 v3, s33, s30, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s31, 0, s33
	v_add_co_u32 v5, s33, s34, v70
	v_add_co_u32 v7, s30, s30, v71
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s33
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	global_load_b64 v[116:117], v[1:2], off offset:40
	global_load_b64 v[112:113], v[3:4], off offset:40
	global_load_b64 v[118:119], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[72:75], v80 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v195 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v195 offset0:2 offset1:3
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
	ds_load_2addr_b64 v[76:79], v195 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
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
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v117, s0
	v_cndmask_b32_e64 v72, 0, v116, s0
	ds_store_2addr_stride64_b64 v186, v[72:73], v[112:113] offset1:16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v119, s1
	v_cndmask_b32_e64 v72, 0, v118, s1
	ds_store_2addr_stride64_b64 v187, v[72:73], v[114:115] offset1:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.11:                               ;   in Loop: Header=BB3_10 Depth=2
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
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[30:31]
	s_add_nc_u64 s[34:35], s[4:5], s[34:35]
	s_lshl_b32 s36, s29, 6
	s_mulk_i32 s8, 0x88
	v_add_co_u32 v72, s33, s30, v68
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[36:37]
	v_add_co_ci_u32_e64 v73, null, s31, 0, s33
	v_add_co_u32 v76, vcc_lo, v76, v194
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_u32 v74, s33, s30, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s30, s34, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	v_add_co_u32 v80, s30, s34, v71
	s_lshl_b32 s8, s29, 2
	v_add_co_ci_u32_e64 v81, null, s35, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	s_clause 0x1
	global_load_b64 v[116:117], v[72:73], off offset:8
	global_load_b64 v[118:119], v[74:75], off offset:8
	s_clause 0x1
	global_load_b64 v[112:113], v[78:79], off offset:8
	global_load_b64 v[114:115], v[80:81], off offset:8
	global_load_b32 v196, v[76:77], off
	global_load_b32 v197, v[82:83], off
.LBB3_12:                               ; %.preheader470.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	s_cselect_b32 s29, s23, 0x3c00
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v195 offset1:1
	ds_load_2addr_stride64_b64 v[80:83], v195 offset0:2 offset1:3
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
	v_add_nc_u32_e32 v76, 0x100, v84
	ds_load_2addr_b64 v[72:75], v195 offset0:32 offset1:96
	ds_load_2addr_stride64_b64 v[76:79], v76 offset0:16 offset1:17
	ds_load_2addr_b64 v[80:83], v195 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[74:75], v[49:56] neg_lo:[0,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[80:81], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[82:83], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[78:79], v[72:73], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[78:79], v[74:75], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[78:79], v[80:81], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[78:79], v[82:83], v[1:8] neg_lo:[0,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v76, s29, v192
	v_add_nc_u32_e32 v72, s26, v193
	s_movk_i32 s26, 0x2000
	s_and_not1_b32 vcc_lo, exec_lo, s8
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
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_9
; %bb.13:                               ; %.preheader471.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_lshrrev_b32_e32 v198, v191, v197
	s_and_b32 s8, s28, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v117, 0, v117, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v189
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s8, s23, 0x3c00
	v_cndmask_b32_e64 v116, 0, v116, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v119, 0, v119, s1
	v_cndmask_b32_e64 v118, 0, v118, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s8, v189
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[116:117], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[118:119], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB3_9
.LBB3_14:                               ; %.preheader468.i
	v_mul_u32_u24_e32 v1, 0x500, v177
	v_lshlrev_b32_e32 v2, 2, v176
	v_lshrrev_b32_e32 v0, 4, v0
	v_mul_u32_u24_e32 v5, 0x50, v176
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add3_u32 v3, 0, v1, v2
	v_and_b32_e32 v2, 15, v0
	v_or_b32_e32 v0, s20, v176
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_u32_u24 v1, 0x280, v167, v3
	v_or_b32_e32 v4, s21, v2
	v_lshlrev_b32_e32 v2, 2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_gt_i32_e64 s7, s12, v0
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	v_ashrrev_i32_e32 v1, 31, v0
	v_add3_u32 v2, 0, v5, v2
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_16
; %bb.15:
	v_mad_co_i64_i32 v[5:6], null, s12, v4, 0
	ds_load_b32 v9, v2
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s0, s16, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s17, v6, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s0, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s0
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off
.LBB3_16:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_18
; %bb.17:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:1280
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off
.LBB3_18:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v6
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_20
; %bb.19:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:2560
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:128
.LBB3_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_22
; %bb.21:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:3840
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:128
.LBB3_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v6
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_24
; %bb.23:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:5120
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:256
.LBB3_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_26
; %bb.25:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	ds_load_b32 v10, v2 offset:6400
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:256
.LBB3_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v6
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_28
; %bb.27:
	v_mad_co_i64_i32 v[6:7], null, s12, v4, 0
	ds_load_b32 v10, v2 offset:7680
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s1, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s1
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:384
.LBB3_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v167
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_30
; %bb.29:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	ds_load_b32 v11, v2 offset:8960
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s1, s16, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s1, v7, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB3_30:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s14, v6
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v175, v174 offset1:20
	ds_store_2addr_b32 v3, v172, v173 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v171, v170 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v169, v168 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_32
; %bb.31:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	ds_load_b32 v11, v2
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s2, s16, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s2, v7, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off
.LBB3_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_105
; %bb.33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_106
.LBB3_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_107
.LBB3_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_108
.LBB3_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_109
.LBB3_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_110
.LBB3_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_40
.LBB3_39:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:8960
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:384
.LBB3_40:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v8, 32, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s3, s14, v8
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v165, v166 offset1:20
	ds_store_2addr_b32 v3, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v159, v160 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_42
; %bb.41:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	ds_load_b32 v13, v2
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s4, s16, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s4, v9, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s4
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB3_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_111
; %bb.43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_112
.LBB3_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_113
.LBB3_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_114
.LBB3_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_115
.LBB3_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_116
.LBB3_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_50
.LBB3_49:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:8960
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB3_50:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v10, 48, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s5, s14, v10
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v157, v158 offset1:20
	ds_store_2addr_b32 v3, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB3_52
; %bb.51:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	ds_load_b32 v15, v2
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s6, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s6, v11, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s6
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB3_52:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_117
; %bb.53:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB3_118
.LBB3_54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_119
.LBB3_55:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_120
.LBB3_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_121
.LBB3_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_122
.LBB3_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_60
.LBB3_59:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:8960
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
.LBB3_60:                               ; %.preheader467.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v12, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s7, s12, v12
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v149, v150 offset1:20
	ds_store_2addr_b32 v3, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_62
; %bb.61:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s8, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s8, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB3_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_64
; %bb.63:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s8, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s8, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s8
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB3_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v12, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v12
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_66
; %bb.65:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s9, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s9, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_66:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_68
; %bb.67:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s9, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s9
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s9, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s9
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_68:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v12, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v12
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_70
; %bb.69:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v16, v2 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v12, s10, s16, v12
	v_add_co_ci_u32_e64 v13, null, s17, v13, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s10, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_70:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_72
; %bb.71:
	v_mad_co_i64_i32 v[12:13], null, s12, v5, 0
	ds_load_b32 v16, v2 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s10, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s10, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s10
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v12, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v12
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB3_74
; %bb.73:
	v_mad_co_i64_i32 v[12:13], null, s12, v4, 0
	ds_load_b32 v4, v2 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v12, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v4, off offset:448
.LBB3_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_76
; %bb.75:
	v_mad_co_i64_i32 v[4:5], null, s12, v5, 0
	ds_load_b32 v14, v2 offset:8960
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:448
.LBB3_76:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v141, v142 offset1:20
	ds_store_2addr_b32 v3, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_123
; %bb.77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_124
.LBB3_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_125
.LBB3_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_126
.LBB3_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_127
.LBB3_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB3_128
.LBB3_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_129
.LBB3_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_85
.LBB3_84:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:8960
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:448
.LBB3_85:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v133, v134 offset1:20
	ds_store_2addr_b32 v3, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_130
; %bb.86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_131
.LBB3_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_132
.LBB3_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_133
.LBB3_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_134
.LBB3_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_135
.LBB3_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_136
.LBB3_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_94
.LBB3_93:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v8, v2 offset:8960
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v8, off offset:448
.LBB3_94:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v125, v126 offset1:20
	ds_store_2addr_b32 v3, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_137
; %bb.95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_138
.LBB3_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_139
.LBB3_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_140
.LBB3_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_141
.LBB3_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_142
.LBB3_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_143
.LBB3_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_103
.LBB3_102:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v5, v2 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[2:3], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v2, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v3, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v5, off offset:448
.LBB3_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_104:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB3_105:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:1280
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_34
.LBB3_106:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:2560
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_35
.LBB3_107:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:3840
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_36
.LBB3_108:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:5120
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_37
.LBB3_109:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	ds_load_b32 v12, v2 offset:6400
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_38
.LBB3_110:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	ds_load_b32 v12, v2 offset:7680
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s3, v8, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s3
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB3_39
	s_branch .LBB3_40
.LBB3_111:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_44
.LBB3_112:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_45
.LBB3_113:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_46
.LBB3_114:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_47
.LBB3_115:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	ds_load_b32 v14, v2 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB3_48
.LBB3_116:
	v_mad_co_i64_i32 v[10:11], null, s12, v8, 0
	ds_load_b32 v14, v2 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s5, v10, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s5
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB3_49
	s_branch .LBB3_50
.LBB3_117:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:1280
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execz .LBB3_54
.LBB3_118:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:2560
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_55
.LBB3_119:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:3840
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_56
.LBB3_120:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:5120
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_57
.LBB3_121:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	ds_load_b32 v16, v2 offset:6400
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB3_58
.LBB3_122:
	v_mad_co_i64_i32 v[12:13], null, s12, v10, 0
	ds_load_b32 v16, v2 offset:7680
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s7, v12, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s7
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB3_59
	s_branch .LBB3_60
.LBB3_123:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_78
.LBB3_124:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:1280
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_79
.LBB3_125:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2 offset:2560
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_80
.LBB3_126:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:3840
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_81
.LBB3_127:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v14, v2 offset:5120
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_82
.LBB3_128:
	v_mad_co_i64_i32 v[4:5], null, s12, v7, 0
	ds_load_b32 v14, v2 offset:6400
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v14, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_83
.LBB3_129:
	v_mad_co_i64_i32 v[4:5], null, s12, v6, 0
	ds_load_b32 v6, v2 offset:7680
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v6, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_84
	s_branch .LBB3_85
.LBB3_130:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_87
.LBB3_131:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:1280
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_88
.LBB3_132:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2 offset:2560
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_89
.LBB3_133:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:3840
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_90
.LBB3_134:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v12, v2 offset:5120
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
.LBB3_135:
	v_mad_co_i64_i32 v[4:5], null, s12, v9, 0
	ds_load_b32 v12, v2 offset:6400
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v12, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_92
.LBB3_136:
	v_mad_co_i64_i32 v[4:5], null, s12, v8, 0
	ds_load_b32 v8, v2 offset:7680
	v_lshlrev_b64_e32 v[6:7], 2, v[0:1]
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s16, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s17, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[4:5], v8, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_93
	s_branch .LBB3_94
.LBB3_137:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_96
.LBB3_138:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:1280
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_97
.LBB3_139:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:2560
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_98
.LBB3_140:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:3840
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_99
.LBB3_141:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:5120
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_100
.LBB3_142:
	v_mad_co_i64_i32 v[3:4], null, s12, v11, 0
	ds_load_b32 v7, v2 offset:6400
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_101
.LBB3_143:
	v_mad_co_i64_i32 v[3:4], null, s12, v10, 0
	ds_load_b32 v7, v2 offset:7680
	v_lshlrev_b64_e32 v[5:6], 2, v[0:1]
	v_lshlrev_b64_e32 v[3:4], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s16, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s17, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, v4, v6, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[3:4], v7, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB3_102
	s_branch .LBB3_103
.Lfunc_end3:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_set, .Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_set
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
; codeLenInByte = 13732
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
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_778842de20dadbdb,@object ; @__hip_cuid_778842de20dadbdb
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_778842de20dadbdb
__hip_cuid_778842de20dadbdb:
	.byte	0                               ; 0x0
	.size	__hip_cuid_778842de20dadbdb, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_778842de20dadbdb
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
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
