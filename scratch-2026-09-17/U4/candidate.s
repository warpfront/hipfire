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
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_lshl_b32 s18, ttmp9, 7
	s_lshl_b32 s20, ttmp7, 7
	v_lshrrev_b32_e32 v120, 5, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s18, s4
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s20, s6
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s24, s0, s1
	s_cmp_eq_u32 s7, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB1_15
; %bb.1:
	s_mov_b32 s25, 0
	s_and_b32 vcc_lo, exec_lo, s24
	s_cbranch_vccz .LBB1_16
; %bb.2:                                ; %.preheader523.i
	s_ashr_i32 s0, s5, 31
	s_add_co_i32 s2, s4, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s5, s0
	s_ashr_i32 s19, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s20, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s6, v5
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_5:                                ; %.preheader518.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s18, v1
	v_lshlrev_b32_e32 v2, 4, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[8:9]
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
.LBB1_6:                                ; %Flow1717
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s12, s6, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s20, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s18, v9
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	v_min_i32_e32 v4, s12, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s12, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v13, 16, v13
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s3, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s3, v3, v[1:2]
	global_load_b64 v[1:2], v68, s[10:11] offset:8
	global_load_b64 v[3:4], v69, s[8:9] offset:8
	global_load_b64 v[5:6], v70, s[10:11] offset:8
	global_load_b64 v[7:8], v71, s[8:9] offset:8
	v_bfe_u32 v12, v0, 1, 1
	v_and_or_b32 v9, v9, 15, v13
	v_or_b32_e32 v14, 8, v120
	v_cmp_gt_i32_e64 s0, s6, v10
	v_mov_b32_e32 v146, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s6, v11
	v_mov_b32_e32 v182, 0
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v168, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v143, 0
	v_bfe_u32 v169, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v156, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_mov_b32 s13, 0
	s_cmp_lt_i32 s5, 0x100
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
; %bb.7:                                ; %.preheader517.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v5, s18, v1
	v_dual_mov_b32 v185, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v7, 6, v169
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v168
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[8:9]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v10, s20, v1
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 2, v6
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 8, v3
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v4, 11, v2
	v_lshl_or_b32 v189, v3, 10, v2
	v_min_i32_e32 v190, s12, v10
	v_cmp_gt_i32_e64 s2, s6, v10
	v_add3_u32 v191, 0, v1, v11
	v_cmp_gt_i32_e64 s3, s4, v5
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v192, 4, v6
	v_add3_u32 v193, 0, v12, v7
	v_add3_u32 v194, 0, v13, v8
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v195, 2, v6
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v135, 0
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
	s_ashr_i32 s7, s6, 31
	s_movk_i32 s29, 0x1000
	s_movk_i32 s21, 0x3000
	s_movk_i32 s26, 0x3800
	s_mov_b32 s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s14, s13
	s_branch .LBB1_9
.LBB1_8:                                ;   in Loop: Header=BB1_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s28
	s_mov_b32 s14, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_9:                                ; %.preheader517.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_11 Depth 2
	s_mov_b32 s15, s13
	s_add_co_i32 s27, s14, 1
	s_mul_u64 s[22:23], s[14:15], 0x88
	s_lshl_b32 s15, s14, 1
	s_cmp_eq_u32 s27, s19
	s_mov_b32 s34, 0
	s_cselect_b32 s28, -1, 0
	s_add_nc_u64 s[22:23], s[8:9], s[22:23]
	s_mov_b32 s31, -1
	s_mov_b32 s33, 0
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=2
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
	s_xor_b32 s12, s31, -1
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
	s_and_b32 vcc_lo, exec_lo, s12
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
	s_cbranch_vccnz .LBB1_8
.LBB1_11:                               ;   Parent Loop BB1_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s12, s34, s15
	s_lshl_b32 s36, s34, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[12:13], s[6:7]
	s_mov_b32 s37, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[38:39], s[38:39], 0x48
	s_add_nc_u64 s[36:37], s[22:23], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[10:11], s[38:39]
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
	s_and_b32 s29, s33, s28
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
	s_cbranch_vccnz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=2
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[12:13], s[6:7]
	s_add_co_i32 s12, s34, s14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_mul_u64 s[38:39], s[12:13], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[10:11], s[36:37]
	s_mulk_i32 s12, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v190, s[36:37]
	v_add_co_u32 v72, s30, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s37, 0, s30
	s_xor_b32 s30, s34, 1
	s_add_nc_u64 s[34:35], s[8:9], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s38, s30, 6
	s_mov_b32 s39, s13
	v_add_co_u32 v76, vcc_lo, v76, v195
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s12
	v_add_co_u32 v74, s40, s36, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s36, s34, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s37, 0, s40
	v_add_co_u32 v80, s34, s34, v71
	s_lshl_b32 s12, s30, 2
	v_add_co_ci_u32_e64 v79, null, s35, 0, s36
	v_add_co_ci_u32_e64 v81, null, s35, 0, s34
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s12
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
.LBB1_13:                               ; %.preheader515.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	v_add_nc_u32_e32 v84, 0, v189
	v_add_nc_u32_e32 v85, 0, v188
	s_xor_b32 s12, s29, -1
	s_and_b32 s29, s31, exec_lo
	s_cselect_b32 s29, s21, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s30, s26, 0x3c00
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
	s_and_not1_b32 vcc_lo, exec_lo, s12
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
	s_cbranch_vccnz .LBB1_10
; %bb.14:                               ; %.preheader516.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s12, s33, exec_lo
	s_cselect_b32 s12, s21, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s12, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s12, s26, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s12, v191
	s_mov_b32 s30, 0
	s_movk_i32 s29, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB1_10
.LBB1_15:
	s_mov_b32 s25, -1
.LBB1_16:                               ; %Flow1740
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB1_178
	s_branch .LBB1_353
.LBB1_17:                               ; %._crit_edge587.i
	v_mul_u32_u24_e32 v1, 0x500, v120
	v_lshlrev_b32_e32 v2, 2, v168
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s23, s4, 31
	s_mov_b32 s22, s4
	s_ashr_i32 s19, s18, 31
	v_add3_u32 v6, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[22:23], s[20:21]
	s_lshl_b64 s[2:3], s[18:19], 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	v_lshrrev_b32_e32 v1, 4, v0
	v_mad_u32_u24 v2, 0x280, v169, v6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	s_add_co_i32 s7, s18, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[12:13], s[0:1], s[2:3]
	ds_store_2addr_b32 v2, v178, v185 offset1:20
	ds_store_2addr_b32 v2, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s13, s13, 0xffff
	s_cmp_gt_i32 s7, s4
	v_mul_lo_u32 v5, s4, v1
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s20, 0x80
	v_lshl_add_u32 v2, v1, 2, 0
	v_or_b32_e32 v4, s20, v1
	v_or_b32_e32 v1, s18, v168
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s6
	s_mov_b32 s15, 0x31004000
	s_cselect_b32 s1, -1, 0
	v_mad_u32_u24 v3, 0x50, v168, v2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s0, s1
	v_cmp_gt_i32_e64 s1, s4, v1
	v_cmp_gt_i32_e64 s0, s6, v4
	s_mov_b32 s14, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s2, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_35
; %bb.18:                               ; %.preheader510.i
	s_and_b32 s7, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s7
	s_cbranch_execz .LBB1_20
; %bb.19:
	v_mad_co_i64_i32 v[7:8], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v9, v3
	global_load_b32 v2, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[7:8], v2, off
.LBB1_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_22
; %bb.21:
	v_mad_co_i64_i32 v[8:9], null, s22, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	ds_load_b32 v10, v3 offset:1280
	global_load_b32 v2, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off
.LBB1_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB1_24
; %bb.23:
	v_mad_co_i64_i32 v[8:9], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s2, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	ds_load_b32 v10, v3 offset:2560
	global_load_b32 v2, v[8:9], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off offset:128
.LBB1_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_26
; %bb.25:
	v_mad_co_i64_i32 v[8:9], null, s22, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	ds_load_b32 v10, v3 offset:3840
	global_load_b32 v2, v[8:9], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off offset:128
.LBB1_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB1_28
; %bb.27:
	v_mad_co_i64_i32 v[8:9], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s2, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	ds_load_b32 v10, v3 offset:5120
	global_load_b32 v2, v[8:9], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off offset:256
.LBB1_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_30
; %bb.29:
	v_mad_co_i64_i32 v[8:9], null, s22, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s1, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	ds_load_b32 v10, v3 offset:6400
	global_load_b32 v2, v[8:9], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off offset:256
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_32
; %bb.31:
	v_mad_co_i64_i32 v[8:9], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s0, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	ds_load_b32 v10, v3 offset:7680
	global_load_b32 v2, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	global_store_b32 v[8:9], v2, off offset:384
.LBB1_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_34
; %bb.33:
	v_mad_co_i64_i32 v[7:8], null, s22, v7, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v2, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	ds_load_b32 v9, v3 offset:8960
	global_load_b32 v2, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[7:8], v2, off offset:384
.LBB1_34:                               ; %Flow1711
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB1_35:                               ; %Flow1712
	v_add_lshl_u32 v5, v5, v168, 2
	v_mul_u32_u24_e32 v2, 0x280, v169
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_37
; %bb.36:                               ; %.preheader.i
	buffer_load_b32 v9, v5, s[12:15], null offen
	ds_load_2addr_stride64_b32 v[7:8], v3 offset1:5
	s_lshl_b32 s0, s4, 8
	s_movk_i32 s1, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s0, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v9
	buffer_store_b32 v7, v5, s[12:15], null offen
	buffer_load_b32 v7, v5, s[12:15], s0 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v7, v8, v7
	buffer_store_b32 v7, v5, s[12:15], s0 offen
	buffer_load_b32 v9, v5, s[12:15], s1 offen
	ds_load_2addr_stride64_b32 v[7:8], v3 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v9
	buffer_store_b32 v7, v5, s[12:15], s1 offen
	buffer_load_b32 v7, v5, s[12:15], s2 offen
	s_movk_i32 s1, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v7, v8, v7
	buffer_store_b32 v7, v5, s[12:15], s2 offen
	buffer_load_b32 v9, v5, s[12:15], s1 offen
	ds_load_2addr_stride64_b32 v[7:8], v3 offset0:20 offset1:25
	s_add_co_i32 s2, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v9
	buffer_store_b32 v7, v5, s[12:15], s1 offen
	buffer_load_b32 v7, v5, s[12:15], s2 offen
	s_movk_i32 s1, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v7, v8, v7
	buffer_store_b32 v7, v5, s[12:15], s2 offen
	buffer_load_b32 v9, v5, s[12:15], s1 offen
	ds_load_2addr_stride64_b32 v[7:8], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v9
	buffer_store_b32 v7, v5, s[12:15], s1 offen
	buffer_load_b32 v7, v5, s[12:15], s0 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v7, v8, v7
	buffer_store_b32 v7, v5, s[12:15], s0 offen
.LBB1_37:                               ; %.loopexit.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v6, v6, v2
	v_cndmask_b32_e64 v7, 0, 1, s3
	v_cmp_gt_i32_e64 s1, s4, v1
	v_or_b32_e32 v9, 16, v4
	s_mov_b32 s0, -1
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v177, v176 offset1:20
	ds_store_2addr_b32 v6, v174, v175 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v173, v172 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v170, v171 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_55
; %bb.38:                               ; %.preheader510.1.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_40
; %bb.39:
	v_mad_co_i64_i32 v[10:11], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s17, v11, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s0, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v8, v13, s0
	ds_load_b32 v8, v3
	global_load_b32 v2, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	global_store_b32 v[10:11], v2, off
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_42
; %bb.41:
	v_mad_co_i64_i32 v[10:11], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v3 offset:1280
	global_load_b32 v2, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off
.LBB1_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_44
; %bb.43:
	v_mad_co_i64_i32 v[10:11], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	ds_load_b32 v12, v3 offset:2560
	global_load_b32 v2, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:128
.LBB1_44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_46
; %bb.45:
	v_mad_co_i64_i32 v[10:11], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v3 offset:3840
	global_load_b32 v2, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:128
.LBB1_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_48
; %bb.47:
	v_mad_co_i64_i32 v[10:11], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	ds_load_b32 v12, v3 offset:5120
	global_load_b32 v2, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:256
.LBB1_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_50
; %bb.49:
	v_mad_co_i64_i32 v[10:11], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v3 offset:6400
	global_load_b32 v2, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:256
.LBB1_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_52
; %bb.51:
	v_mad_co_i64_i32 v[10:11], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, v2, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	ds_load_b32 v12, v3 offset:7680
	global_load_b32 v2, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:384
.LBB1_52:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_54
; %bb.53:
	v_mad_co_i64_i32 v[10:11], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, v2, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v8, v13, vcc_lo
	ds_load_b32 v8, v3 offset:8960
	global_load_b32 v2, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	global_store_b32 v[10:11], v2, off offset:384
.LBB1_54:                               ; %Flow1709
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_55:                               ; %Flow1710
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_57
; %bb.56:                               ; %.preheader.1.i
	s_lshl_b32 s0, s4, 6
	ds_load_2addr_stride64_b32 v[10:11], v3 offset1:5
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_mul_i32 s1, s4, 0x140
	s_add_co_i32 s2, s0, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s1, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s2, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_add_co_i32 s3, s1, 0x100
	s_addk_co_i32 s1, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
.LBB1_57:                               ; %.loopexit.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	v_cmp_gt_i32_e64 s1, s4, v1
	v_or_b32_e32 v8, 32, v4
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v166, v167 offset1:20
	ds_store_2addr_b32 v6, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_75
; %bb.58:                               ; %.preheader510.2.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_60
; %bb.59:
	v_mad_co_i64_i32 v[10:11], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s0, v2, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s0
	ds_load_b32 v12, v3
	global_load_b32 v2, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off
.LBB1_60:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v10
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_62
; %bb.61:
	v_mad_co_i64_i32 v[11:12], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v3 offset:1280
	global_load_b32 v2, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off
.LBB1_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_64
; %bb.63:
	v_mad_co_i64_i32 v[11:12], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v3 offset:2560
	global_load_b32 v2, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:128
.LBB1_64:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_co_i64_i32 v[11:12], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v3 offset:3840
	global_load_b32 v2, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:128
.LBB1_66:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_68
; %bb.67:
	v_mad_co_i64_i32 v[11:12], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v3 offset:5120
	global_load_b32 v2, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:256
.LBB1_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_70
; %bb.69:
	v_mad_co_i64_i32 v[11:12], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v3 offset:6400
	global_load_b32 v2, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:256
.LBB1_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_72
; %bb.71:
	v_mad_co_i64_i32 v[11:12], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	ds_load_b32 v13, v3 offset:7680
	global_load_b32 v2, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:384
.LBB1_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_74
; %bb.73:
	v_mad_co_i64_i32 v[10:11], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s17, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, v2, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	ds_load_b32 v12, v3 offset:8960
	global_load_b32 v2, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[10:11], v2, off offset:384
.LBB1_74:                               ; %Flow1707
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_75:                               ; %Flow1708
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_77
; %bb.76:                               ; %.preheader.2.i
	s_lshl_b32 s0, s4, 7
	ds_load_2addr_stride64_b32 v[10:11], v3 offset1:5
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_mul_i32 s1, s4, 0x180
	s_add_co_i32 s2, s0, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s1, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s2, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_add_co_i32 s3, s1, 0x100
	s_addk_co_i32 s1, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	ds_load_2addr_stride64_b32 v[10:11], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v10, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
.LBB1_77:                               ; %.loopexit.2.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	v_or_b32_e32 v10, 48, v4
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v158, v159 offset1:20
	ds_store_2addr_b32 v6, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_95
; %bb.78:                               ; %.preheader510.3.i
	v_cmp_gt_i32_e64 s1, s4, v1
	v_cmp_gt_i32_e32 vcc_lo, s6, v10
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_80
; %bb.79:
	v_mad_co_i64_i32 v[11:12], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s0, v2, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	ds_load_b32 v13, v3
	global_load_b32 v2, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off
.LBB1_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v11
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_82
; %bb.81:
	v_mad_co_i64_i32 v[12:13], null, s22, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:1280
	global_load_b32 v2, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off
.LBB1_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_84
; %bb.83:
	v_mad_co_i64_i32 v[12:13], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v3 offset:2560
	global_load_b32 v2, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:128
.LBB1_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_86
; %bb.85:
	v_mad_co_i64_i32 v[12:13], null, s22, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:3840
	global_load_b32 v2, v[12:13], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:128
.LBB1_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_88
; %bb.87:
	v_mad_co_i64_i32 v[12:13], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v3 offset:5120
	global_load_b32 v2, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:256
.LBB1_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_90
; %bb.89:
	v_mad_co_i64_i32 v[12:13], null, s22, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:6400
	global_load_b32 v2, v[12:13], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:256
.LBB1_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_92
; %bb.91:
	v_mad_co_i64_i32 v[12:13], null, s22, v10, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v2, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	ds_load_b32 v14, v3 offset:7680
	global_load_b32 v2, v[12:13], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:384
.LBB1_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_94
; %bb.93:
	v_mad_co_i64_i32 v[11:12], null, s22, v11, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s17, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v2, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	ds_load_b32 v13, v3 offset:8960
	global_load_b32 v2, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[11:12], v2, off offset:384
.LBB1_94:                               ; %Flow1705
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_95:                               ; %Flow1706
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_97
; %bb.96:                               ; %.preheader.3.i
	s_mul_i32 s0, s4, 0xc0
	ds_load_2addr_stride64_b32 v[11:12], v3 offset1:5
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_mul_i32 s1, s4, 0x1c0
	s_add_co_i32 s2, s0, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s3, s1, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v12, v2
	ds_load_2addr_stride64_b32 v[11:12], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s2, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v12, v2
	ds_load_2addr_stride64_b32 v[11:12], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_add_co_i32 s3, s1, 0x100
	s_addk_co_i32 s1, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v12, v2
	ds_load_2addr_stride64_b32 v[11:12], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v11, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
.LBB1_97:                               ; %.loopexit.3.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	v_or_b32_e32 v11, 16, v1
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v150, v151 offset1:20
	ds_store_2addr_b32 v6, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v147, v146 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_115
; %bb.98:                               ; %.preheader510.1662.i
	v_cmp_gt_i32_e64 s1, s4, v11
	v_cmp_gt_i32_e32 vcc_lo, s6, v4
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_100
; %bb.99:
	v_mad_co_i64_i32 v[12:13], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s0, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s0
	ds_load_b32 v14, v3
	global_load_b32 v2, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:64
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v12, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v12
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_102
; %bb.101:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:1280
	global_load_b32 v2, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:64
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_104
; %bb.103:
	v_mad_co_i64_i32 v[13:14], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s2, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s2
	ds_load_b32 v15, v3 offset:2560
	global_load_b32 v2, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:192
.LBB1_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_106
; %bb.105:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:3840
	global_load_b32 v2, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:192
.LBB1_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_108
; %bb.107:
	v_mad_co_i64_i32 v[13:14], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s2, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s2
	ds_load_b32 v15, v3 offset:5120
	global_load_b32 v2, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:320
.LBB1_108:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_110
; %bb.109:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:6400
	global_load_b32 v2, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:320
.LBB1_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_112
; %bb.111:
	v_mad_co_i64_i32 v[13:14], null, s22, v4, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, vcc_lo, v2, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, vcc_lo
	ds_load_b32 v15, v3 offset:7680
	global_load_b32 v2, v[13:14], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:448
.LBB1_112:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_114
; %bb.113:
	v_mad_co_i64_i32 v[12:13], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v2, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	ds_load_b32 v14, v3 offset:8960
	global_load_b32 v2, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:448
.LBB1_114:                              ; %Flow1703
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_115:                              ; %Flow1704
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_117
; %bb.116:                              ; %.preheader.1691.i
	s_mov_b32 s0, 64
	ds_load_2addr_stride64_b32 v[12:13], v3 offset1:5
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_lshl_b32 s1, s4, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s1, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_movk_i32 s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_or_b32 s2, s1, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_movk_i32 s0, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_add_co_i32 s2, s1, 0x140
	s_addk_co_i32 s1, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_movk_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s2 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
.LBB1_117:                              ; %.loopexit.1700.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v142, v143 offset1:20
	ds_store_2addr_b32 v6, v140, v141 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v138, v139 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v136, v137 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_135
; %bb.118:                              ; %.preheader510.1.1.i
	v_cmp_gt_i32_e64 s1, s4, v11
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_120
; %bb.119:
	v_mad_co_i64_i32 v[12:13], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s0, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s0
	ds_load_b32 v14, v3
	global_load_b32 v2, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:64
.LBB1_120:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v12, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v12
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_122
; %bb.121:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:1280
	global_load_b32 v2, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:64
.LBB1_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_124
; %bb.123:
	v_mad_co_i64_i32 v[13:14], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s2, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s2
	ds_load_b32 v15, v3 offset:2560
	global_load_b32 v2, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:192
.LBB1_124:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_126
; %bb.125:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:3840
	global_load_b32 v2, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:192
.LBB1_126:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_128
; %bb.127:
	v_mad_co_i64_i32 v[13:14], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s2, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s2
	ds_load_b32 v15, v3 offset:5120
	global_load_b32 v2, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:320
.LBB1_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_130
; %bb.129:
	v_mad_co_i64_i32 v[13:14], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, s17, v14, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, s1, v2, v15
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s1
	ds_load_b32 v15, v3 offset:6400
	global_load_b32 v2, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v15, v2
	global_store_b32 v[13:14], v2, off offset:320
.LBB1_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_132
; %bb.131:
	v_mad_co_i64_i32 v[13:14], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v13, vcc_lo, v2, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, v9, v16, vcc_lo
	ds_load_b32 v9, v3 offset:7680
	global_load_b32 v2, v[13:14], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[13:14], v2, off offset:448
.LBB1_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_134
; %bb.133:
	v_mad_co_i64_i32 v[12:13], null, s22, v12, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v2, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v9, v15, vcc_lo
	ds_load_b32 v9, v3 offset:8960
	global_load_b32 v2, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[12:13], v2, off offset:448
.LBB1_134:                              ; %Flow1701
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_135:                              ; %Flow1702
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_137
; %bb.136:                              ; %.preheader.1.1.i
	s_lshl_b32 s0, s4, 6
	ds_load_2addr_stride64_b32 v[12:13], v3 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s0, 64
	s_mul_i32 s2, s4, 0x140
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 0x140
	s_addk_co_i32 s2, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	ds_load_2addr_stride64_b32 v[12:13], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v13, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
.LBB1_137:                              ; %.loopexit.1.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v134, v135 offset1:20
	ds_store_2addr_b32 v6, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v130, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v128, v129 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_155
; %bb.138:                              ; %.preheader510.2.1.i
	v_cmp_gt_i32_e64 s1, s4, v11
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_140
; %bb.139:
	v_mad_co_i64_i32 v[12:13], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s0, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s17, v13, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s0, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v9, v15, s0
	ds_load_b32 v9, v3
	global_load_b32 v2, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[12:13], v2, off offset:64
.LBB1_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v9
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_142
; %bb.141:
	v_mad_co_i64_i32 v[12:13], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:1280
	global_load_b32 v2, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:64
.LBB1_142:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_144
; %bb.143:
	v_mad_co_i64_i32 v[12:13], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v3 offset:2560
	global_load_b32 v2, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:192
.LBB1_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_146
; %bb.145:
	v_mad_co_i64_i32 v[12:13], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:3840
	global_load_b32 v2, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:192
.LBB1_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_148
; %bb.147:
	v_mad_co_i64_i32 v[12:13], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s2, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v3 offset:5120
	global_load_b32 v2, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:320
.LBB1_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_150
; %bb.149:
	v_mad_co_i64_i32 v[12:13], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, s1, s16, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s17, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v2, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v3 offset:6400
	global_load_b32 v2, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v14, v2
	global_store_b32 v[12:13], v2, off offset:320
.LBB1_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v2, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v2
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_152
; %bb.151:
	v_mad_co_i64_i32 v[12:13], null, s22, v8, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s17, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v2, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v8, v15, vcc_lo
	ds_load_b32 v8, v3 offset:7680
	global_load_b32 v2, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	global_store_b32 v[12:13], v2, off offset:448
.LBB1_152:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_154
; %bb.153:
	v_mad_co_i64_i32 v[8:9], null, s22, v9, 0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, s16, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v2, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v13, vcc_lo
	ds_load_b32 v12, v3 offset:8960
	global_load_b32 v2, v[8:9], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v12, v2
	global_store_b32 v[8:9], v2, off offset:448
.LBB1_154:                              ; %Flow1699
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_155:                              ; %Flow1700
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_157
; %bb.156:                              ; %.preheader.2.1.i
	s_lshl_b32 s0, s4, 7
	ds_load_2addr_stride64_b32 v[8:9], v3 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 64
	s_mul_i32 s2, s4, 0x180
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_or_b32 s3, s2, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v9, v2
	ds_load_2addr_stride64_b32 v[8:9], v3 offset0:10 offset1:15
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v9, v2
	ds_load_2addr_stride64_b32 v[8:9], v3 offset0:20 offset1:25
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 0x140
	s_addk_co_i32 s2, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	buffer_store_b32 v2, v5, s[12:15], s1 offen
	buffer_load_b32 v2, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v9, v2
	ds_load_2addr_stride64_b32 v[8:9], v3 offset0:30 offset1:35
	buffer_store_b32 v2, v5, s[12:15], s3 offen
	buffer_load_b32 v2, v5, s[12:15], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v2, v8, v2
	buffer_store_b32 v2, v5, s[12:15], s0 offen
	buffer_load_b32 v2, v5, s[12:15], s2 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v2, v9, v2
	buffer_store_b32 v2, v5, s[12:15], s2 offen
.LBB1_157:                              ; %.loopexit.2.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v7
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v6, v126, v127 offset1:20
	ds_store_2addr_b32 v6, v124, v125 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v122, v123 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v121, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_175
; %bb.158:                              ; %.preheader510.3.1.i
	v_cmp_gt_i32_e64 s1, s4, v11
	v_cmp_gt_i32_e32 vcc_lo, s6, v10
	v_ashrrev_i32_e32 v2, 31, v1
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_160
; %bb.159:
	v_mad_co_i64_i32 v[6:7], null, s22, v10, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s0, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s0
	v_add_co_u32 v6, s0, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s0
	ds_load_b32 v9, v3
	global_load_b32 v8, v[6:7], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:64
.LBB1_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v4, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v4
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_162
; %bb.161:
	v_mad_co_i64_i32 v[6:7], null, s22, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
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
	ds_load_b32 v9, v3 offset:1280
	global_load_b32 v8, v[6:7], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:64
.LBB1_162:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_164
; %bb.163:
	v_mad_co_i64_i32 v[6:7], null, s22, v10, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s2, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s2
	v_add_co_u32 v6, s2, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s2
	ds_load_b32 v9, v3 offset:2560
	global_load_b32 v8, v[6:7], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:192
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_166
; %bb.165:
	v_mad_co_i64_i32 v[6:7], null, s22, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
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
	ds_load_b32 v9, v3 offset:3840
	global_load_b32 v8, v[6:7], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:192
.LBB1_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_168
; %bb.167:
	v_mad_co_i64_i32 v[6:7], null, s22, v10, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, s2, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s2
	v_add_co_u32 v6, s2, v6, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s2
	ds_load_b32 v9, v3 offset:5120
	global_load_b32 v8, v[6:7], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:320
.LBB1_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_170
; %bb.169:
	v_mad_co_i64_i32 v[6:7], null, s22, v4, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
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
	ds_load_b32 v9, v3 offset:6400
	global_load_b32 v8, v[6:7], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:320
.LBB1_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_172
; %bb.171:
	v_mad_co_i64_i32 v[6:7], null, s22, v10, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, s16, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	ds_load_b32 v9, v3 offset:7680
	global_load_b32 v8, v[6:7], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v9, v8
	global_store_b32 v[6:7], v8, off offset:448
.LBB1_172:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_174
; %bb.173:
	v_mad_co_i64_i32 v[6:7], null, s22, v4, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v4, vcc_lo, s16, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s17, v7, vcc_lo
	v_add_co_u32 v1, vcc_lo, v4, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v6, v2, vcc_lo
	ds_load_b32 v6, v3 offset:8960
	global_load_b32 v4, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v6, v4
	global_store_b32 v[1:2], v4, off offset:448
.LBB1_174:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_175:                              ; %Flow1698
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_177
; %bb.176:                              ; %.preheader.3.1.i
	s_mul_i32 s0, s4, 0xc0
	ds_load_2addr_stride64_b32 v[1:2], v3 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s0, 64
	s_mul_i32 s2, s4, 0x1c0
	buffer_load_b32 v4, v5, s[12:15], s1 offen
	s_add_co_i32 s3, s2, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v4
	buffer_store_b32 v1, v5, s[12:15], s1 offen
	buffer_load_b32 v1, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v5, s[12:15], s3 offen
	buffer_load_b32 v4, v5, s[12:15], s1 offen
	ds_load_2addr_stride64_b32 v[1:2], v3 offset0:10 offset1:15
	s_add_co_i32 s3, s2, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v4
	buffer_store_b32 v1, v5, s[12:15], s1 offen
	buffer_load_b32 v1, v5, s[12:15], s3 offen
	s_add_co_i32 s1, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v5, s[12:15], s3 offen
	buffer_load_b32 v4, v5, s[12:15], s1 offen
	ds_load_2addr_stride64_b32 v[1:2], v3 offset0:20 offset1:25
	s_add_co_i32 s3, s2, 0x140
	s_addk_co_i32 s2, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v4
	buffer_store_b32 v1, v5, s[12:15], s1 offen
	buffer_load_b32 v1, v5, s[12:15], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v5, s[12:15], s3 offen
	buffer_load_b32 v4, v5, s[12:15], s0 offen
	ds_load_2addr_stride64_b32 v[1:2], v3 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v1, v4
	buffer_store_b32 v1, v5, s[12:15], s0 offen
	buffer_load_b32 v1, v5, s[12:15], s2 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v2, v1
	buffer_store_b32 v1, v5, s[12:15], s2 offen
.LBB1_177:                              ; %.loopexit.3.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
	s_and_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_353
.LBB1_178:
	s_and_b32 vcc_lo, exec_lo, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_353
; %bb.179:                              ; %.preheader518.i
	s_ashr_i32 s0, s5, 31
	s_add_co_i32 s2, s4, -1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b32 s0, s0, 24
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s5, s0
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s19, s0, 8
	s_mov_b32 s0, exec_lo
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s3, s19, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_183
; %bb.180:                              ; %.lr.ph.i121
	v_lshrrev_b32_e32 v1, 1, v0
	v_and_b32_e32 v2, 1, v0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v5, s20, v1
	v_lshlrev_b32_e32 v3, 2, v2
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s6, v5
	s_cbranch_execz .LBB1_182
; %bb.181:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_182:                              ; %.preheader513.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s18, v1
	v_lshlrev_b32_e32 v2, 4, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[8:9]
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
.LBB1_183:                              ; %Flow1739
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s12, s6, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s20, v9
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v2, s18, v9
	v_dual_mov_b32 v152, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v126, 0 :: v_dual_add_nc_u32 v3, 64, v2
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s12, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s12, v11
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v174, 0 :: v_dual_and_b32 v13, 16, v13
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[68:69], null, 0x48, v4, v[1:2]
	v_mad_co_u64_u32 v[69:70], null, s3, v2, v[1:2]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[71:72], null, s3, v3, v[1:2]
	global_load_b64 v[1:2], v68, s[10:11] offset:8
	global_load_b64 v[3:4], v69, s[8:9] offset:8
	global_load_b64 v[5:6], v70, s[10:11] offset:8
	global_load_b64 v[7:8], v71, s[8:9] offset:8
	v_bfe_u32 v12, v0, 1, 1
	v_and_or_b32 v9, v9, 15, v13
	v_or_b32_e32 v14, 8, v120
	v_cmp_gt_i32_e64 s0, s6, v10
	v_mov_b32_e32 v146, 0
	v_and_or_b32 v13, v120, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s6, v11
	v_mov_b32_e32 v182, 0
	v_dual_mov_b32 v123, 0 :: v_dual_and_b32 v168, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v143, 0
	v_bfe_u32 v169, v0, 4, 1
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v156, 0
	v_add_nc_u32_e32 v186, 0, v13
	v_add_nc_u32_e32 v187, 0, v9
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_mov_b32 s13, 0
	s_cmp_lt_i32 s5, 0x100
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
	s_cbranch_scc1 .LBB1_192
; %bb.184:                              ; %.preheader512.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v5, s18, v1
	v_dual_mov_b32 v185, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v7, 6, v169
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v8, 3, v168
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[8:9]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v10, s20, v1
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v11, 2, v6
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v12, 8, v3
	v_dual_mov_b32 v180, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v188, v4, 11, v2
	v_lshl_or_b32 v189, v3, 10, v2
	v_min_i32_e32 v190, s12, v10
	v_cmp_gt_i32_e64 s2, s6, v10
	v_add3_u32 v191, 0, v1, v11
	v_cmp_gt_i32_e64 s3, s4, v5
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v192, 4, v6
	v_add3_u32 v193, 0, v12, v7
	v_add3_u32 v194, 0, v13, v8
	v_dual_mov_b32 v150, 0 :: v_dual_lshlrev_b32 v195, 2, v6
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v135, 0
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
	s_ashr_i32 s7, s6, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s5, 0x3000
	s_movk_i32 s21, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s14, s13
	s_branch .LBB1_186
.LBB1_185:                              ;   in Loop: Header=BB1_186 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s14, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_192
.LBB1_186:                              ; %.preheader512.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_188 Depth 2
	s_mov_b32 s15, s13
	s_add_co_i32 s24, s14, 1
	s_mul_u64 s[22:23], s[14:15], 0x88
	s_lshl_b32 s15, s14, 1
	s_cmp_eq_u32 s24, s19
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[8:9], s[22:23]
	s_mov_b32 s28, -1
	s_mov_b32 s29, 0
	s_branch .LBB1_188
.LBB1_187:                              ;   in Loop: Header=BB1_188 Depth=2
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
	s_xor_b32 s12, s28, -1
	s_mov_b32 s30, 1
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
	s_mov_b32 s28, 0
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
	s_and_b32 vcc_lo, exec_lo, s12
	s_mov_b32 s29, -1
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
	s_cbranch_vccnz .LBB1_185
.LBB1_188:                              ;   Parent Loop BB1_186 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s12, s30, s15
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[12:13], s[6:7]
	s_mov_b32 s35, s13
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[10:11], s[36:37]
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
	s_cbranch_vccnz .LBB1_190
; %bb.189:                              ;   in Loop: Header=BB1_188 Depth=2
	s_add_co_i32 s12, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[12:13], s[6:7]
	s_add_co_i32 s12, s30, s14
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[12:13], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[10:11], s[34:35]
	s_mulk_i32 s12, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v190, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[8:9], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s13
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s12
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s12, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s12
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
.LBB1_190:                              ; %.preheader510.i51
                                        ;   in Loop: Header=BB1_188 Depth=2
	v_add_nc_u32_e32 v84, 0, v189
	v_add_nc_u32_e32 v85, 0, v188
	s_xor_b32 s12, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s5, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s21, 0x3c00
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
	s_and_not1_b32 vcc_lo, exec_lo, s12
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
	s_cbranch_vccnz .LBB1_187
; %bb.191:                              ; %.preheader511.i
                                        ;   in Loop: Header=BB1_188 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s12, s29, exec_lo
	s_cselect_b32 s12, s5, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s12, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s12, s21, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s12, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB1_187
.LBB1_192:                              ; %._crit_edge582.i
	v_mul_u32_u24_e32 v1, 0x500, v120
	v_lshlrev_b32_e32 v2, 2, v168
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s5, s4, 31
	s_ashr_i32 s19, s18, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[4:5], s[20:21]
	v_add3_u32 v5, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[18:19], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	v_lshrrev_b32_e32 v0, 4, v0
	v_mad_u32_u24 v1, 0x280, v169, v5
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[0:1], s[2:3]
	s_add_co_i32 s5, s18, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s9, s9, 0xffff
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_cmp_gt_i32 s5, s4
	v_lshl_add_u32 v1, v0, 2, 0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s20, 0x80
	v_mul_lo_u32 v4, s4, v0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s6
	v_or_b32_e32 v3, s20, v0
	s_cselect_b32 s1, -1, 0
	v_or_b32_e32 v0, s18, v168
	v_mad_u32_u24 v2, 0x50, v168, v1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s0, s1
	s_mov_b32 s11, 0x31004000
	s_mov_b32 s10, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s0, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_210
; %bb.193:                              ; %.preheader505.i
	v_cmp_gt_i32_e64 s1, s4, v0
	v_cmp_gt_i32_e32 vcc_lo, s6, v3
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_195
; %bb.194:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
	ds_load_b32 v10, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, s0, s16, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s0
	v_add_co_u32 v6, s0, v1, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s0
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off
.LBB1_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v6
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_197
; %bb.196:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off
.LBB1_197:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB1_199
; %bb.198:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s2, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s2
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:128
.LBB1_199:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_201
; %bb.200:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:128
.LBB1_201:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB1_203
; %bb.202:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s2, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s2
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:256
.LBB1_203:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_205
; %bb.204:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:256
.LBB1_205:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s5, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s5
	s_cbranch_execz .LBB1_207
; %bb.206:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB1_207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_209
; %bb.208:
	v_mad_co_i64_i32 v[6:7], null, s4, v6, 0
	ds_load_b32 v10, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, vcc_lo, s16, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s17, v7, vcc_lo
	v_add_co_u32 v6, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:384
.LBB1_209:                              ; %Flow1733
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_210:                              ; %Flow1734
	v_add_lshl_u32 v4, v4, v168, 2
	v_mul_u32_u24_e32 v1, 0x280, v169
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_212
; %bb.211:                              ; %.preheader.i38
	ds_load_2addr_stride64_b32 v[6:7], v2 offset1:5
	ds_load_2addr_stride64_b32 v[8:9], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[10:11], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[12:13], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 8
	s_movk_i32 s1, 0x80
	s_movk_i32 s2, 0x100
	s_movk_i32 s5, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s7, s0, 0x80
	s_add_co_i32 s12, s0, 0x100
	s_add_co_i32 s13, s0, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v6, v4, s[8:11], null offen
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	buffer_store_b32 v9, v4, s[8:11], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v10, v4, s[8:11], s2 offen
	buffer_store_b32 v11, v4, s[8:11], s12 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v12, v4, s[8:11], s5 offen
	buffer_store_b32 v13, v4, s[8:11], s13 offen
.LBB1_212:                              ; %.loopexit.i25
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v5, v1
	v_cndmask_b32_e64 v6, 0, 1, s3
	s_mov_b32 s0, -1
	s_and_not1_b32 vcc_lo, exec_lo, s3
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
	s_cbranch_vccnz .LBB1_230
; %bb.213:                              ; %.preheader505.1.i
	v_or_b32_e32 v7, 16, v3
	v_cmp_gt_i32_e64 s1, s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_215
; %bb.214:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off
.LBB1_215:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_217
; %bb.216:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB1_217:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_219
; %bb.218:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_219:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_221
; %bb.220:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_221:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_223
; %bb.222:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_223:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_225
; %bb.224:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_225:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_227
; %bb.226:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:384
.LBB1_227:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_229
; %bb.228:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB1_229:                              ; %Flow1731
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_230:                              ; %Flow1732
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_232
; %bb.231:                              ; %.preheader.1.i37
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 6
	s_mul_i32 s1, s4, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 0x80
	s_add_co_i32 s3, s1, 0x80
	s_add_co_i32 s5, s0, 0x100
	s_add_co_i32 s7, s1, 0x100
	s_add_co_i32 s12, s0, 0x180
	s_add_co_i32 s13, s1, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s2 offen
	buffer_store_b32 v10, v4, s[8:11], s3 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s5 offen
	buffer_store_b32 v12, v4, s[8:11], s7 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s12 offen
	buffer_store_b32 v14, v4, s[8:11], s13 offen
.LBB1_232:                              ; %.loopexit.1.i26
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v166, v167 offset1:20
	ds_store_2addr_b32 v5, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_250
; %bb.233:                              ; %.preheader505.2.i
	v_or_b32_e32 v7, 32, v3
	v_cmp_gt_i32_e64 s1, s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_235
; %bb.234:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off
.LBB1_235:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_237
; %bb.236:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB1_237:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_239
; %bb.238:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_239:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_241
; %bb.240:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_241:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_243
; %bb.242:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_243:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_245
; %bb.244:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_245:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_247
; %bb.246:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:384
.LBB1_247:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_249
; %bb.248:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB1_249:                              ; %Flow1729
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_250:                              ; %Flow1730
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_252
; %bb.251:                              ; %.preheader.2.i36
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 7
	s_mul_i32 s1, s4, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 0x80
	s_add_co_i32 s3, s1, 0x80
	s_add_co_i32 s5, s0, 0x100
	s_add_co_i32 s7, s1, 0x100
	s_add_co_i32 s12, s0, 0x180
	s_add_co_i32 s13, s1, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s2 offen
	buffer_store_b32 v10, v4, s[8:11], s3 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s5 offen
	buffer_store_b32 v12, v4, s[8:11], s7 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s12 offen
	buffer_store_b32 v14, v4, s[8:11], s13 offen
.LBB1_252:                              ; %.loopexit.2.i27
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v158, v159 offset1:20
	ds_store_2addr_b32 v5, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_270
; %bb.253:                              ; %.preheader505.3.i
	v_or_b32_e32 v7, 48, v3
	v_cmp_gt_i32_e64 s1, s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_255
; %bb.254:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off
.LBB1_255:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_257
; %bb.256:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB1_257:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_259
; %bb.258:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_259:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_261
; %bb.260:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB1_261:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_263
; %bb.262:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_263:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_265
; %bb.264:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB1_265:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_267
; %bb.266:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:384
.LBB1_267:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_269
; %bb.268:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB1_269:                              ; %Flow1727
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_270:                              ; %Flow1728
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_272
; %bb.271:                              ; %.preheader.3.i35
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_mul_i32 s0, s4, 0xc0
	s_mul_i32 s1, s4, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 0x80
	s_add_co_i32 s3, s1, 0x80
	s_add_co_i32 s5, s0, 0x100
	s_add_co_i32 s7, s1, 0x100
	s_add_co_i32 s12, s0, 0x180
	s_add_co_i32 s13, s1, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s2 offen
	buffer_store_b32 v10, v4, s[8:11], s3 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s5 offen
	buffer_store_b32 v12, v4, s[8:11], s7 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s12 offen
	buffer_store_b32 v14, v4, s[8:11], s13 offen
.LBB1_272:                              ; %.loopexit.3.i28
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v150, v151 offset1:20
	ds_store_2addr_b32 v5, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v147, v146 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_290
; %bb.273:                              ; %.preheader505.1657.i
	v_or_b32_e32 v1, 16, v0
	v_cmp_gt_i32_e32 vcc_lo, s6, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_275
; %bb.274:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s0, s16, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, s0
	v_add_co_u32 v7, s0, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s0
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:64
.LBB1_275:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_277
; %bb.276:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	v_add_co_u32 v8, s1, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:64
.LBB1_277:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_279
; %bb.278:
	v_mad_co_i64_i32 v[8:9], null, s4, v3, 0
	ds_load_b32 v12, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s2, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s2
	v_add_co_u32 v8, s2, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:192
.LBB1_279:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_281
; %bb.280:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	v_add_co_u32 v8, s1, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:192
.LBB1_281:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_283
; %bb.282:
	v_mad_co_i64_i32 v[8:9], null, s4, v3, 0
	ds_load_b32 v12, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s2, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s2
	v_add_co_u32 v8, s2, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:320
.LBB1_283:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_285
; %bb.284:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s1, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s1
	v_add_co_u32 v8, s1, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:320
.LBB1_285:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_287
; %bb.286:
	v_mad_co_i64_i32 v[8:9], null, s4, v3, 0
	ds_load_b32 v12, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, vcc_lo, s16, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, vcc_lo
	v_add_co_u32 v8, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:448
.LBB1_287:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_289
; %bb.288:
	v_mad_co_i64_i32 v[7:8], null, s4, v7, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448
.LBB1_289:                              ; %Flow1725
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_290:                              ; %Flow1726
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_292
; %bb.291:                              ; %.preheader.1686.i
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_lshl_b32 s1, s4, 8
	s_mov_b32 s0, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s7, s1, 64
	s_movk_i32 s2, 0xc0
	s_movk_i32 s3, 0x140
	s_movk_i32 s5, 0x1c0
	s_or_b32 s12, s1, 0xc0
	s_add_co_i32 s13, s1, 0x140
	s_addk_co_i32 s1, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	buffer_store_b32 v8, v4, s[8:11], s7 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s2 offen
	buffer_store_b32 v10, v4, s[8:11], s12 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s3 offen
	buffer_store_b32 v12, v4, s[8:11], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s5 offen
	buffer_store_b32 v14, v4, s[8:11], s1 offen
.LBB1_292:                              ; %.loopexit.1695.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
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
	s_cbranch_vccnz .LBB1_310
; %bb.293:                              ; %.preheader505.1.1.i
	v_or_b32_e32 v1, 16, v0
	v_or_b32_e32 v7, 16, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s4, v1
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_295
; %bb.294:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:64
.LBB1_295:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_297
; %bb.296:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
.LBB1_297:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_299
; %bb.298:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
.LBB1_299:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_301
; %bb.300:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
.LBB1_301:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_303
; %bb.302:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
.LBB1_303:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_305
; %bb.304:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
.LBB1_305:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_307
; %bb.306:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:448
.LBB1_307:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_309
; %bb.308:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448
.LBB1_309:                              ; %Flow1723
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_310:                              ; %Flow1724
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_312
; %bb.311:                              ; %.preheader.1.1.i34
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 6
	s_mul_i32 s1, s4, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 64
	s_add_co_i32 s3, s1, 64
	s_add_co_i32 s5, s0, 0xc0
	s_add_co_i32 s7, s1, 0xc0
	s_add_co_i32 s12, s0, 0x140
	s_add_co_i32 s13, s1, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_addk_co_i32 s1, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s2 offen
	buffer_store_b32 v8, v4, s[8:11], s3 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s5 offen
	buffer_store_b32 v10, v4, s[8:11], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s12 offen
	buffer_store_b32 v12, v4, s[8:11], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s0 offen
	buffer_store_b32 v14, v4, s[8:11], s1 offen
.LBB1_312:                              ; %.loopexit.1.1.i29
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
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
	s_cbranch_vccnz .LBB1_330
; %bb.313:                              ; %.preheader505.2.1.i
	v_or_b32_e32 v1, 16, v0
	v_or_b32_e32 v7, 32, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s4, v1
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_315
; %bb.314:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	ds_load_b32 v12, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v1, s0, s16, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s17, v9, s0
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v12, off offset:64
.LBB1_315:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_317
; %bb.316:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:64
.LBB1_317:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_319
; %bb.318:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
.LBB1_319:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_321
; %bb.320:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:192
.LBB1_321:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_323
; %bb.322:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
.LBB1_323:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_325
; %bb.324:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s16, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:320
.LBB1_325:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_327
; %bb.326:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s16, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s17, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:448
.LBB1_327:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_329
; %bb.328:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v11, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s17, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:448
.LBB1_329:                              ; %Flow1721
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_330:                              ; %Flow1722
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_332
; %bb.331:                              ; %.preheader.2.1.i33
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 7
	s_mul_i32 s1, s4, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s0, 64
	s_or_b32 s3, s1, 64
	s_add_co_i32 s5, s0, 0xc0
	s_add_co_i32 s7, s1, 0xc0
	s_add_co_i32 s12, s0, 0x140
	s_add_co_i32 s13, s1, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_addk_co_i32 s1, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s2 offen
	buffer_store_b32 v8, v4, s[8:11], s3 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s5 offen
	buffer_store_b32 v10, v4, s[8:11], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s12 offen
	buffer_store_b32 v12, v4, s[8:11], s13 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s0 offen
	buffer_store_b32 v14, v4, s[8:11], s1 offen
.LBB1_332:                              ; %.loopexit.2.1.i30
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
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
	s_cbranch_vccnz .LBB1_350
; %bb.333:                              ; %.preheader505.3.1.i
	v_or_b32_e32 v1, 16, v0
	v_or_b32_e32 v5, 48, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s1, s4, v1
	v_cmp_gt_i32_e32 vcc_lo, s6, v5
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB1_335
; %bb.334:
	v_mad_co_i64_i32 v[6:7], null, s4, v5, 0
	ds_load_b32 v10, v2
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s0, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s0, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s0
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:64
.LBB1_335:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v3, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_337
; %bb.336:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
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
	global_store_b32 v[6:7], v10, off offset:64
.LBB1_337:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_339
; %bb.338:
	v_mad_co_i64_i32 v[6:7], null, s4, v5, 0
	ds_load_b32 v10, v2 offset:2560
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s2, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s2, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s2
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:192
.LBB1_339:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_341
; %bb.340:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
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
	global_store_b32 v[6:7], v10, off offset:192
.LBB1_341:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_343
; %bb.342:
	v_mad_co_i64_i32 v[6:7], null, s4, v5, 0
	ds_load_b32 v10, v2 offset:5120
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s2, s16, v6
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s17, v7, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, s2, v6, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s2
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:320
.LBB1_343:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_345
; %bb.344:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
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
	global_store_b32 v[6:7], v10, off offset:320
.LBB1_345:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v6
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_347
; %bb.346:
	v_mad_co_i64_i32 v[5:6], null, s4, v5, 0
	ds_load_b32 v9, v2 offset:7680
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s16, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:448
.LBB1_347:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_349
; %bb.348:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	ds_load_b32 v3, v2 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s16, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s17, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v5, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v6, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v3, off offset:448
.LBB1_349:                              ; %Flow1718
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB1_350:                              ; %Flow1720
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_352
; %bb.351:                              ; %.preheader.3.1.i32
	ds_load_2addr_stride64_b32 v[0:1], v2 offset1:5
	ds_load_2addr_stride64_b32 v[5:6], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[7:8], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[2:3], v2 offset0:30 offset1:35
	s_mul_i32 s0, s4, 0xc0
	s_mul_i32 s1, s4, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, 64
	s_add_co_i32 s3, s1, 64
	s_add_co_i32 s4, s0, 0xc0
	s_add_co_i32 s5, s1, 0xc0
	s_add_co_i32 s6, s0, 0x140
	s_add_co_i32 s7, s1, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_addk_co_i32 s1, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v4, s[8:11], s2 offen
	buffer_store_b32 v1, v4, s[8:11], s3 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v5, v4, s[8:11], s4 offen
	buffer_store_b32 v6, v4, s[8:11], s5 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s6 offen
	buffer_store_b32 v8, v4, s[8:11], s7 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v2, v4, s[8:11], s0 offen
	buffer_store_b32 v3, v4, s[8:11], s1 offen
.LBB1_352:                              ; %.loopexit.3.1.i31
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
.LBB1_353:                              ; %Flow1742
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_355
; %bb.354:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_355:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
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
; codeLenInByte = 33968
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
	s_load_b96 s[4:6], s[0:1], 0x18
	s_lshl_b32 s14, ttmp9, 7
	s_lshl_b32 s16, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s14, s4
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s16, s6
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_175
; %bb.1:                                ; %.preheader523.i
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x10
	s_ashr_i32 s0, s5, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s5, s0
	s_add_co_i32 s2, s4, -1
	s_ashr_i32 s15, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s15, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s16, v2
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v3, 2, v1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s6, v5
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB2_4:                                ; %.preheader518.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s14, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[8:9]
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
.LBB2_5:                                ; %Flow806
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s17, s6, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s16, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s14, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v6, s17, v12
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s17, v13
	v_min_i32_e32 v5, s2, v5
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s3, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s3, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[10:11] offset:8
	global_load_b64 v[5:6], v69, s[8:9] offset:8
	global_load_b64 v[7:8], v70, s[10:11] offset:8
	global_load_b64 v[9:10], v71, s[8:9] offset:8
	v_lshrrev_b32_e32 v177, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s6, v12
	v_mov_b32_e32 v148, 0
	v_or_b32_e32 v15, 8, v177
	v_and_or_b32 v16, v177, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s6, v13
	v_mov_b32_e32 v182, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v152, 0 :: v_dual_and_b32 v167, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v141, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v168, v0, 4, 1
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
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_mov_b32 s19, 0
	s_cmp_lt_i32 s5, 0x100
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
; %bb.6:                                ; %.preheader517.lr.ph.i
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v3, s14, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s2, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v185, 0 :: v_dual_add_nc_u32 v8, s16, v2
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_mad_co_u64_u32 v[65:66], null, s3, v7, s[8:9]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 2, v1
	v_min_i32_e32 v188, s17, v8
	v_cmp_gt_i32_e64 s2, s6, v8
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v5
	v_lshl_or_b32 v189, v6, 11, v4
	v_lshl_or_b32 v190, v5, 10, v4
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v4, 6, v168
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v5, 9, v6
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v6, 3, v167
	v_mad_co_u64_u32 v[66:67], null, s3, v7, v[66:67]
	v_add3_u32 v191, 0, v2, v9
	v_cmp_gt_i32_e64 s3, s4, v3
	v_dual_mov_b32 v145, 0 :: v_dual_lshlrev_b32 v192, 4, v1
	v_add3_u32 v193, 0, v8, v4
	v_add3_u32 v194, 0, v5, v6
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v195, 2, v1
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s7, s6, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s5, 0x3000
	s_movk_i32 s17, 0x3800
	s_mov_b32 s27, 0
	s_mov_b32 s20, s19
	s_branch .LBB2_8
.LBB2_7:                                ;   in Loop: Header=BB2_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s20, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_8:                                ; %.preheader517.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_10 Depth 2
	s_mov_b32 s21, s19
	s_add_co_i32 s24, s20, 1
	s_mul_u64 s[22:23], s[20:21], 0x88
	s_lshl_b32 s21, s20, 1
	s_cmp_eq_u32 s24, s15
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[22:23], s[8:9], s[22:23]
	s_mov_b32 s28, -1
	s_mov_b32 s29, 0
	s_branch .LBB2_10
.LBB2_9:                                ;   in Loop: Header=BB2_10 Depth=2
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
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_cvt_f32_i32_e32 v41, v41
	v_add_f32_e32 v172, v172, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v170, v170, v52 :: v_dual_add_f32 v169, v169, v54
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v43
	v_cvt_f32_i32_e32 v44, v44
	v_dual_mul_f32 v50, v107, v90 :: v_dual_add_f32 v165, v165, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v42, v52, v51
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v41, v49, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v44, v105, v90 :: v_dual_fmac_f32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v46, v46
	v_dual_add_f32 v166, v166, v42 :: v_dual_fmac_f32 v41, v44, v51
	s_delay_alu instid0(VALU_DEP_3)
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
	s_wait_dscnt 0x8
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
	s_xor_b32 s18, s28, -1
	s_mov_b32 s30, 1
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s18
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
	s_cbranch_vccnz .LBB2_7
.LBB2_10:                               ;   Parent Loop BB2_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s18, s30, s21
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[18:19], s[6:7]
	s_mov_b32 s35, s19
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[10:11], s[36:37]
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
	s_cbranch_vccnz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[18:19], s[6:7]
	s_add_co_i32 s18, s30, s20
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[18:19], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[10:11], s[34:35]
	s_mulk_i32 s18, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[8:9], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s19
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s18
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s18, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s18
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
.LBB2_12:                               ; %.preheader515.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	v_add_nc_u32_e32 v85, 0, v189
	s_xor_b32 s18, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s5, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s17, 0x3c00
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
	s_and_not1_b32 vcc_lo, exec_lo, s18
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
	s_cbranch_vccnz .LBB2_9
; %bb.13:                               ; %.preheader516.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s18, s29, exec_lo
	s_cselect_b32 s18, s5, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s18, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s18, s17, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s18, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB2_9
.LBB2_14:                               ; %._crit_edge587.i
	v_mul_u32_u24_e32 v1, 0x500, v177
	v_lshlrev_b32_e32 v2, 2, v167
	s_ashr_i32 s17, s16, 31
	s_ashr_i32 s5, s4, 31
	s_ashr_i32 s15, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[4:5], s[16:17]
	v_add3_u32 v5, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[14:15], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[12:13], s[0:1]
	v_lshrrev_b32_e32 v0, 4, v0
	v_mad_u32_u24 v1, 0x280, v168, v5
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[0:1], s[2:3]
	s_add_co_i32 s5, s14, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s9, s9, 0xffff
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_cmp_gt_i32 s5, s4
	v_mul_lo_u32 v4, s4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s16, 0x80
	v_lshl_add_u32 v1, v0, 2, 0
	v_or_b32_e32 v3, s16, v0
	v_or_b32_e32 v0, s14, v167
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s6
	s_mov_b32 s11, 0x31004000
	s_cselect_b32 s1, -1, 0
	v_mad_u32_u24 v2, 0x50, v167, v1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s0, s1
	v_cmp_gt_i32_e64 s1, s4, v0
	v_cmp_gt_i32_e64 s0, s6, v3
	s_mov_b32 s10, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s2, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_32
; %bb.15:                               ; %.preheader510.i
	s_and_b32 s5, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s5
	s_cbranch_execz .LBB2_17
; %bb.16:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s13, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	ds_load_b32 v8, v2
	global_load_b32 v1, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	global_store_b32 v[6:7], v1, off
.LBB2_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s6, v6
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_19
; %bb.18:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v2 offset:1280
	global_load_b32 v1, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off
.LBB2_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_21
; %bb.20:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	ds_load_b32 v9, v2 offset:2560
	global_load_b32 v1, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off offset:128
.LBB2_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_23
; %bb.22:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v2 offset:3840
	global_load_b32 v1, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off offset:128
.LBB2_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_25
; %bb.24:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	ds_load_b32 v9, v2 offset:5120
	global_load_b32 v1, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off offset:256
.LBB2_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_27
; %bb.26:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v2 offset:6400
	global_load_b32 v1, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off offset:256
.LBB2_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_29
; %bb.28:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, s0, v1, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s0
	ds_load_b32 v9, v2 offset:7680
	global_load_b32 v1, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	global_store_b32 v[7:8], v1, off offset:384
.LBB2_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_31
; %bb.30:
	v_mad_co_i64_i32 v[6:7], null, s4, v6, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s13, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	ds_load_b32 v8, v2 offset:8960
	global_load_b32 v1, v[6:7], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	global_store_b32 v[6:7], v1, off offset:384
.LBB2_31:                               ; %Flow800
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB2_32:                               ; %Flow801
	v_add_lshl_u32 v4, v4, v167, 2
	v_mul_u32_u24_e32 v1, 0x280, v168
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_34
; %bb.33:                               ; %.preheader.i
	buffer_load_b32 v8, v4, s[8:11], null offen
	ds_load_2addr_stride64_b32 v[6:7], v2 offset1:5
	s_lshl_b32 s0, s4, 8
	s_movk_i32 s1, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s0, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v8
	buffer_store_b32 v6, v4, s[8:11], null offen
	buffer_load_b32 v6, v4, s[8:11], s0 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v6, v7, v6
	buffer_store_b32 v6, v4, s[8:11], s0 offen
	buffer_load_b32 v8, v4, s[8:11], s1 offen
	ds_load_2addr_stride64_b32 v[6:7], v2 offset0:10 offset1:15
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v8
	buffer_store_b32 v6, v4, s[8:11], s1 offen
	buffer_load_b32 v6, v4, s[8:11], s2 offen
	s_movk_i32 s1, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v6, v7, v6
	buffer_store_b32 v6, v4, s[8:11], s2 offen
	buffer_load_b32 v8, v4, s[8:11], s1 offen
	ds_load_2addr_stride64_b32 v[6:7], v2 offset0:20 offset1:25
	s_add_co_i32 s2, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v8
	buffer_store_b32 v6, v4, s[8:11], s1 offen
	buffer_load_b32 v6, v4, s[8:11], s2 offen
	s_movk_i32 s1, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v6, v7, v6
	buffer_store_b32 v6, v4, s[8:11], s2 offen
	buffer_load_b32 v8, v4, s[8:11], s1 offen
	ds_load_2addr_stride64_b32 v[6:7], v2 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v8
	buffer_store_b32 v6, v4, s[8:11], s1 offen
	buffer_load_b32 v6, v4, s[8:11], s0 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v6, v7, v6
	buffer_store_b32 v6, v4, s[8:11], s0 offen
.LBB2_34:                               ; %.loopexit.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v5, v1
	v_cndmask_b32_e64 v6, 0, 1, s3
	v_cmp_gt_i32_e64 s1, s4, v0
	v_or_b32_e32 v7, 16, v3
	s_mov_b32 s0, -1
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v176, v175 offset1:20
	ds_store_2addr_b32 v5, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v172, v171 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v170, v169 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_52
; %bb.35:                               ; %.preheader510.1.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_37
; %bb.36:
	v_mad_co_i64_i32 v[8:9], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, s13, v9, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, s0, v1, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s0
	ds_load_b32 v10, v2
	global_load_b32 v1, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	global_store_b32 v[8:9], v1, off
.LBB2_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_39
; %bb.38:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	ds_load_b32 v11, v2 offset:1280
	global_load_b32 v1, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off
.LBB2_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_41
; %bb.40:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	ds_load_b32 v11, v2 offset:2560
	global_load_b32 v1, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:128
.LBB2_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_43
; %bb.42:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	ds_load_b32 v11, v2 offset:3840
	global_load_b32 v1, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:128
.LBB2_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_45
; %bb.44:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	ds_load_b32 v11, v2 offset:5120
	global_load_b32 v1, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:256
.LBB2_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_47
; %bb.46:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	ds_load_b32 v11, v2 offset:6400
	global_load_b32 v1, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:256
.LBB2_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_49
; %bb.48:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v11, v2 offset:7680
	global_load_b32 v1, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:384
.LBB2_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_51
; %bb.50:
	v_mad_co_i64_i32 v[8:9], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[10:11], 2, v[0:1]
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s13, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v8, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, v9, v11, vcc_lo
	ds_load_b32 v10, v2 offset:8960
	global_load_b32 v1, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	global_store_b32 v[8:9], v1, off offset:384
.LBB2_51:                               ; %Flow798
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_52:                               ; %Flow799
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s3, s4, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_54
; %bb.53:                               ; %.preheader.1.i
	s_lshl_b32 s0, s4, 6
	ds_load_2addr_stride64_b32 v[8:9], v2 offset1:5
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_add_co_i32 s1, s0, 0x80
	s_add_co_i32 s2, s3, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v9, v1
	ds_load_2addr_stride64_b32 v[8:9], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s3 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v9, v1
	ds_load_2addr_stride64_b32 v[8:9], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s3, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s3, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v9, v1
	ds_load_2addr_stride64_b32 v[8:9], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
.LBB2_54:                               ; %.loopexit.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_cmp_gt_i32_e64 s1, s4, v0
	v_or_b32_e32 v8, 32, v3
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v165, v166 offset1:20
	ds_store_2addr_b32 v5, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v159, v160 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_72
; %bb.55:                               ; %.preheader510.2.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_57
; %bb.56:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, s0, v1, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	ds_load_b32 v11, v2
	global_load_b32 v1, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off
.LBB2_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v9, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v9
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_59
; %bb.58:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v2 offset:1280
	global_load_b32 v1, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off
.LBB2_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_61
; %bb.60:
	v_mad_co_i64_i32 v[10:11], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	ds_load_b32 v12, v2 offset:2560
	global_load_b32 v1, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:128
.LBB2_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_63
; %bb.62:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v2 offset:3840
	global_load_b32 v1, v[10:11], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:128
.LBB2_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB2_65
; %bb.64:
	v_mad_co_i64_i32 v[10:11], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s2, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	ds_load_b32 v12, v2 offset:5120
	global_load_b32 v1, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:256
.LBB2_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_67
; %bb.66:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	ds_load_b32 v12, v2 offset:6400
	global_load_b32 v1, v[10:11], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:256
.LBB2_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s5, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s5
	s_cbranch_execz .LBB2_69
; %bb.68:
	v_mad_co_i64_i32 v[10:11], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	ds_load_b32 v12, v2 offset:7680
	global_load_b32 v1, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:384
.LBB2_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_71
; %bb.70:
	v_mad_co_i64_i32 v[9:10], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v9
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	ds_load_b32 v11, v2 offset:8960
	global_load_b32 v1, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[9:10], v1, off offset:384
.LBB2_71:                               ; %Flow796
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_72:                               ; %Flow797
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s5, s4, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_74
; %bb.73:                               ; %.preheader.2.i
	s_lshl_b32 s0, s4, 7
	ds_load_2addr_stride64_b32 v[9:10], v2 offset1:5
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_add_co_i32 s1, s0, 0x80
	s_add_co_i32 s2, s5, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v10, v1
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s5 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0x100
	s_addk_co_i32 s0, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v10, v1
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s5, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s5, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v10, v1
	ds_load_2addr_stride64_b32 v[9:10], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v9, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
.LBB2_74:                               ; %.loopexit.2.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_or_b32_e32 v9, 48, v3
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v157, v158 offset1:20
	ds_store_2addr_b32 v5, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_92
; %bb.75:                               ; %.preheader510.3.i
	v_cmp_gt_i32_e64 s1, s4, v0
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_77
; %bb.76:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, s0, v1, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s0
	ds_load_b32 v12, v2
	global_load_b32 v1, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off
.LBB2_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v10
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_79
; %bb.78:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:1280
	global_load_b32 v1, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off
.LBB2_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB2_81
; %bb.80:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v2 offset:2560
	global_load_b32 v1, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:128
.LBB2_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_83
; %bb.82:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:3840
	global_load_b32 v1, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:128
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB2_85
; %bb.84:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v2 offset:5120
	global_load_b32 v1, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:256
.LBB2_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_87
; %bb.86:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:6400
	global_load_b32 v1, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:256
.LBB2_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s7, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s7
	s_cbranch_execz .LBB2_89
; %bb.88:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	ds_load_b32 v13, v2 offset:7680
	global_load_b32 v1, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:384
.LBB2_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_91
; %bb.90:
	v_mad_co_i64_i32 v[10:11], null, s4, v10, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s13, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v10, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	ds_load_b32 v12, v2 offset:8960
	global_load_b32 v1, v[10:11], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v12, v1
	global_store_b32 v[10:11], v1, off offset:384
.LBB2_91:                               ; %Flow794
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_92:                               ; %Flow795
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s14, s4, 0xc0
	s_mul_i32 s7, s4, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_94
; %bb.93:                               ; %.preheader.3.i
	buffer_load_b32 v1, v4, s[8:11], s14 offen
	ds_load_2addr_stride64_b32 v[10:11], v2 offset1:5
	s_add_co_i32 s0, s14, 0x80
	s_add_co_i32 s1, s7, 0x80
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v4, s[8:11], s14 offen
	buffer_load_b32 v1, v4, s[8:11], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v11, v1
	ds_load_2addr_stride64_b32 v[10:11], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s7 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s0, s14, 0x100
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v11, v1
	ds_load_2addr_stride64_b32 v[10:11], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_add_co_i32 s1, s7, 0x100
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s0, s14, 0x180
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v11, v1
	ds_load_2addr_stride64_b32 v[10:11], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_add_co_i32 s1, s7, 0x180
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v10, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
.LBB2_94:                               ; %.loopexit.3.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_or_b32_e32 v10, 16, v0
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v149, v150 offset1:20
	ds_store_2addr_b32 v5, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_112
; %bb.95:                               ; %.preheader510.1662.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v3
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_97
; %bb.96:
	v_mad_co_i64_i32 v[11:12], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	ds_load_b32 v13, v2
	global_load_b32 v1, v[11:12], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:64
.LBB2_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v11
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_99
; %bb.98:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:1280
	global_load_b32 v1, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:64
.LBB2_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB2_101
; %bb.100:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v2 offset:2560
	global_load_b32 v1, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:192
.LBB2_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_103
; %bb.102:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:3840
	global_load_b32 v1, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:192
.LBB2_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB2_105
; %bb.104:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v2 offset:5120
	global_load_b32 v1, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:320
.LBB2_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_107
; %bb.106:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:6400
	global_load_b32 v1, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:320
.LBB2_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s15, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s15
	s_cbranch_execz .LBB2_109
; %bb.108:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	ds_load_b32 v14, v2 offset:7680
	global_load_b32 v1, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:448
.LBB2_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_111
; %bb.110:
	v_mad_co_i64_i32 v[11:12], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	ds_load_b32 v13, v2 offset:8960
	global_load_b32 v1, v[11:12], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:448
.LBB2_111:                              ; %Flow792
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_112:                              ; %Flow793
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_114
; %bb.113:                              ; %.preheader.1691.i
	s_mov_b32 s0, 64
	ds_load_2addr_stride64_b32 v[11:12], v2 offset1:5
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_lshl_b32 s1, s4, 8
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s2, s1, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_movk_i32 s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_or_b32 s2, s1, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_movk_i32 s0, 0x140
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_add_co_i32 s2, s1, 0x140
	s_addk_co_i32 s1, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_movk_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
.LBB2_114:                              ; %.loopexit.1700.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v141, v142 offset1:20
	ds_store_2addr_b32 v5, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_132
; %bb.115:                              ; %.preheader510.1.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_117
; %bb.116:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	ds_load_b32 v13, v2
	global_load_b32 v1, v[11:12], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:64
.LBB2_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v11
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_119
; %bb.118:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:1280
	global_load_b32 v1, v[12:13], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:64
.LBB2_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB2_121
; %bb.120:
	v_mad_co_i64_i32 v[12:13], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v2 offset:2560
	global_load_b32 v1, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:192
.LBB2_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_123
; %bb.122:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:3840
	global_load_b32 v1, v[12:13], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:192
.LBB2_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB2_125
; %bb.124:
	v_mad_co_i64_i32 v[12:13], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	ds_load_b32 v14, v2 offset:5120
	global_load_b32 v1, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:320
.LBB2_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_127
; %bb.126:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	ds_load_b32 v14, v2 offset:6400
	global_load_b32 v1, v[12:13], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v14, v1
	global_store_b32 v[12:13], v1, off offset:320
.LBB2_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s15, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s15
	s_cbranch_execz .LBB2_129
; %bb.128:
	v_mad_co_i64_i32 v[12:13], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v12
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s13, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v12, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v13, null, v7, v15, vcc_lo
	ds_load_b32 v7, v2 offset:7680
	global_load_b32 v1, v[12:13], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	global_store_b32 v[12:13], v1, off offset:448
.LBB2_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_131
; %bb.130:
	v_mad_co_i64_i32 v[11:12], null, s4, v11, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s13, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v7, v14, vcc_lo
	ds_load_b32 v7, v2 offset:8960
	global_load_b32 v1, v[11:12], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	global_store_b32 v[11:12], v1, off offset:448
.LBB2_131:                              ; %Flow790
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_132:                              ; %Flow791
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_134
; %bb.133:                              ; %.preheader.1.1.i
	s_lshl_b32 s0, s4, 6
	ds_load_2addr_stride64_b32 v[11:12], v2 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s1, s0, 64
	s_add_co_i32 s2, s3, 64
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s3, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s3, 0x140
	s_addk_co_i32 s3, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s3 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v12, v1
	buffer_store_b32 v1, v4, s[8:11], s3 offen
.LBB2_134:                              ; %.loopexit.1.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v133, v134 offset1:20
	ds_store_2addr_b32 v5, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_152
; %bb.135:                              ; %.preheader510.2.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_137
; %bb.136:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s13, v12, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v7, v14, s0
	ds_load_b32 v7, v2
	global_load_b32 v1, v[11:12], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	global_store_b32 v[11:12], v1, off offset:64
.LBB2_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_139
; %bb.138:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:1280
	global_load_b32 v1, v[11:12], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:64
.LBB2_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_141
; %bb.140:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v2 offset:2560
	global_load_b32 v1, v[11:12], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:192
.LBB2_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_143
; %bb.142:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:3840
	global_load_b32 v1, v[11:12], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:192
.LBB2_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_145
; %bb.144:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	ds_load_b32 v13, v2 offset:5120
	global_load_b32 v1, v[11:12], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:320
.LBB2_145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_147
; %bb.146:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	ds_load_b32 v13, v2 offset:6400
	global_load_b32 v1, v[11:12], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v13, v1
	global_store_b32 v[11:12], v1, off offset:320
.LBB2_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_149
; %bb.148:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s13, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, v8, v14, vcc_lo
	ds_load_b32 v8, v2 offset:7680
	global_load_b32 v1, v[11:12], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v8, v1
	global_store_b32 v[11:12], v1, off offset:448
.LBB2_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_151
; %bb.150:
	v_mad_co_i64_i32 v[7:8], null, s4, v7, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s13, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, v8, v12, vcc_lo
	ds_load_b32 v11, v2 offset:8960
	global_load_b32 v1, v[7:8], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v11, v1
	global_store_b32 v[7:8], v1, off offset:448
.LBB2_151:                              ; %Flow788
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_152:                              ; %Flow789
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_154
; %bb.153:                              ; %.preheader.2.1.i
	s_lshl_b32 s0, s4, 7
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, 64
	s_or_b32 s2, s5, 64
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v8, v1
	ds_load_2addr_stride64_b32 v[7:8], v2 offset0:10 offset1:15
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s5, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_add_co_i32 s1, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v8, v1
	ds_load_2addr_stride64_b32 v[7:8], v2 offset0:20 offset1:25
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s1 offen
	s_add_co_i32 s2, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	buffer_load_b32 v1, v4, s[8:11], s2 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v8, v1
	ds_load_2addr_stride64_b32 v[7:8], v2 offset0:30 offset1:35
	buffer_store_b32 v1, v4, s[8:11], s2 offen
	buffer_load_b32 v1, v4, s[8:11], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v7, v1
	buffer_store_b32 v1, v4, s[8:11], s0 offen
	buffer_load_b32 v1, v4, s[8:11], s5 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v1, v8, v1
	buffer_store_b32 v1, v4, s[8:11], s5 offen
.LBB2_154:                              ; %.loopexit.2.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v125, v126 offset1:20
	ds_store_2addr_b32 v5, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_172
; %bb.155:                              ; %.preheader510.3.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB2_157
; %bb.156:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s0, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s0
	v_add_co_u32 v5, s0, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s0
	ds_load_b32 v8, v2
	global_load_b32 v7, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:64
.LBB2_157:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v3, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_159
; %bb.158:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	ds_load_b32 v8, v2 offset:1280
	global_load_b32 v7, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:64
.LBB2_159:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_161
; %bb.160:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s2, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s2
	v_add_co_u32 v5, s2, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s2
	ds_load_b32 v8, v2 offset:2560
	global_load_b32 v7, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:192
.LBB2_161:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_163
; %bb.162:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	ds_load_b32 v8, v2 offset:3840
	global_load_b32 v7, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:192
.LBB2_163:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_165
; %bb.164:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s2, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s2
	v_add_co_u32 v5, s2, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s2
	ds_load_b32 v8, v2 offset:5120
	global_load_b32 v7, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:320
.LBB2_165:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_167
; %bb.166:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	ds_load_b32 v8, v2 offset:6400
	global_load_b32 v7, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:320
.LBB2_167:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_169
; %bb.168:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v5, vcc_lo, s12, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s13, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v8, v2 offset:7680
	global_load_b32 v7, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v8, v7
	global_store_b32 v[5:6], v7, off offset:448
.LBB2_169:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_171
; %bb.170:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v3, vcc_lo, s12, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s13, v6, vcc_lo
	v_add_co_u32 v0, vcc_lo, v3, v0
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v5, v1, vcc_lo
	ds_load_b32 v5, v2 offset:8960
	global_load_b32 v3, v[0:1], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v3, v5, v3
	global_store_b32 v[0:1], v3, off offset:448
.LBB2_171:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB2_172:                              ; %Flow787
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_174
; %bb.173:                              ; %.preheader.3.1.i
	s_add_co_i32 s0, s14, 64
	ds_load_2addr_stride64_b32 v[0:1], v2 offset1:5
	buffer_load_b32 v3, v4, s[8:11], s0 offen
	s_add_co_i32 s1, s7, 64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	buffer_store_b32 v0, v4, s[8:11], s0 offen
	buffer_load_b32 v0, v4, s[8:11], s1 offen
	s_add_co_i32 s0, s14, 0xc0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v4, s[8:11], s1 offen
	buffer_load_b32 v3, v4, s[8:11], s0 offen
	ds_load_2addr_stride64_b32 v[0:1], v2 offset0:10 offset1:15
	s_add_co_i32 s1, s7, 0xc0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	buffer_store_b32 v0, v4, s[8:11], s0 offen
	buffer_load_b32 v0, v4, s[8:11], s1 offen
	s_add_co_i32 s0, s14, 0x140
	s_addk_co_i32 s14, 0x1c0
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v4, s[8:11], s1 offen
	buffer_load_b32 v3, v4, s[8:11], s0 offen
	ds_load_2addr_stride64_b32 v[0:1], v2 offset0:20 offset1:25
	s_add_co_i32 s1, s7, 0x140
	s_addk_co_i32 s7, 0x1c0
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	buffer_store_b32 v0, v4, s[8:11], s0 offen
	buffer_load_b32 v0, v4, s[8:11], s1 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v4, s[8:11], s1 offen
	buffer_load_b32 v3, v4, s[8:11], s14 offen
	ds_load_2addr_stride64_b32 v[0:1], v2 offset0:30 offset1:35
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v0, v3
	buffer_store_b32 v0, v4, s[8:11], s14 offen
	buffer_load_b32 v0, v4, s[8:11], s7 offen
	s_wait_loadcnt 0x0
	v_add_f32_e32 v0, v1, v0
	buffer_store_b32 v0, v4, s[8:11], s7 offen
.LBB2_174:                              ; %.loopexit.3.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_175:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
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
; codeLenInByte = 17960
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
	s_load_b96 s[4:6], s[0:1], 0x18
	s_lshl_b32 s14, ttmp9, 7
	s_lshl_b32 s16, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s14, s4
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s16, s6
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_175
; %bb.1:                                ; %.preheader518.i
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x10
	s_ashr_i32 s0, s5, 31
	v_lshrrev_b32_e32 v2, 1, v0
	s_lshr_b32 s0, s0, 24
	v_and_b32_e32 v1, 1, v0
	s_add_co_i32 s0, s5, s0
	s_add_co_i32 s2, s4, -1
	s_ashr_i32 s15, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s3, s15, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s16, v2
	v_dual_mov_b32 v4, 0 :: v_dual_lshlrev_b32 v3, 2, v1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s6, v5
	s_cbranch_execz .LBB3_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[10:11]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v3
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB3_4:                                ; %.preheader513.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s14, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s4, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s3, v5, s[8:9]
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
.LBB3_5:                                ; %Flow806
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s17, s6, -1
	v_dual_mov_b32 v154, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s16, v11
	v_dual_mov_b32 v121, 0 :: v_dual_add_nc_u32 v4, s14, v11
	v_dual_mov_b32 v126, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v122, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v124, 0 :: v_dual_add_nc_u32 v5, 64, v4
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v6, s17, v12
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s17, v13
	v_min_i32_e32 v5, s2, v5
	v_dual_mov_b32 v176, 0 :: v_dual_and_b32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s3, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s3, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[10:11] offset:8
	global_load_b64 v[5:6], v69, s[8:9] offset:8
	global_load_b64 v[7:8], v70, s[10:11] offset:8
	global_load_b64 v[9:10], v71, s[8:9] offset:8
	v_lshrrev_b32_e32 v177, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s6, v12
	v_mov_b32_e32 v148, 0
	v_or_b32_e32 v15, 8, v177
	v_and_or_b32 v16, v177, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s6, v13
	v_mov_b32_e32 v182, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v152, 0 :: v_dual_and_b32 v167, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v141, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v168, v0, 4, 1
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
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v169, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v184, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v178, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v185, 0
	s_mov_b32 s19, 0
	s_cmp_lt_i32 s5, 0x100
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
; %bb.6:                                ; %.preheader512.lr.ph.i
	v_dual_mov_b32 v178, 0 :: v_dual_add_nc_u32 v3, s14, v2
	v_dual_mov_b32 v197, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s2, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v185, 0 :: v_dual_add_nc_u32 v8, s16, v2
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_mad_co_u64_u32 v[65:66], null, s3, v7, s[8:9]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v183, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v196, 0 :: v_dual_lshlrev_b32 v9, 2, v1
	v_min_i32_e32 v188, s17, v8
	v_cmp_gt_i32_e64 s2, s6, v8
	v_dual_mov_b32 v179, 0 :: v_dual_lshlrev_b32 v8, 8, v5
	v_lshl_or_b32 v189, v6, 11, v4
	v_lshl_or_b32 v190, v5, 10, v4
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v4, 6, v168
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v5, 9, v6
	v_dual_mov_b32 v147, 0 :: v_dual_lshlrev_b32 v6, 3, v167
	v_mad_co_u64_u32 v[66:67], null, s3, v7, v[66:67]
	v_add3_u32 v191, 0, v2, v9
	v_cmp_gt_i32_e64 s3, s4, v3
	v_dual_mov_b32 v145, 0 :: v_dual_lshlrev_b32 v192, 4, v1
	v_add3_u32 v193, 0, v8, v4
	v_add3_u32 v194, 0, v5, v6
	v_dual_mov_b32 v182, 0 :: v_dual_lshlrev_b32 v195, 2, v1
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v120, 0
	s_ashr_i32 s7, s6, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s5, 0x3000
	s_movk_i32 s17, 0x3800
	s_mov_b32 s27, 0
	s_mov_b32 s20, s19
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s20, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_8:                                ; %.preheader512.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_10 Depth 2
	s_mov_b32 s21, s19
	s_add_co_i32 s24, s20, 1
	s_mul_u64 s[22:23], s[20:21], 0x88
	s_lshl_b32 s21, s20, 1
	s_cmp_eq_u32 s24, s15
	s_mov_b32 s30, 0
	s_cselect_b32 s25, -1, 0
	s_add_nc_u64 s[22:23], s[8:9], s[22:23]
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
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v49, v110, v90 :: v_dual_fmac_f32 v54, v55, v59
	v_cvt_f32_i32_e32 v41, v41
	v_add_f32_e32 v172, v172, v50
	v_mul_f32_e32 v50, v108, v90
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v51, v91
	v_mul_f32_e32 v41, v49, v41
	v_mul_f32_e32 v49, v111, v90
	v_cvt_f32_i32_e32 v43, v43
	v_mul_f32_e32 v42, v50, v42
	v_mul_f32_e32 v50, v106, v90
	v_dual_add_f32 v170, v170, v52 :: v_dual_add_f32 v169, v169, v54
	v_fmac_f32_e32 v41, v49, v51
	v_dual_mul_f32 v49, v104, v90 :: v_dual_mul_f32 v52, v109, v90
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v43, v50, v43
	v_cvt_f32_i32_e32 v44, v44
	v_dual_mul_f32 v50, v107, v90 :: v_dual_add_f32 v165, v165, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v42, v52, v51
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v41, v49, v44
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v44, v105, v90 :: v_dual_fmac_f32 v43, v50, v51
	v_dual_mul_f32 v49, v100, v90 :: v_dual_mul_f32 v50, v102, v90
	v_cvt_f32_i32_e32 v46, v46
	v_dual_add_f32 v166, v166, v42 :: v_dual_fmac_f32 v41, v44, v51
	s_delay_alu instid0(VALU_DEP_3)
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
	s_wait_dscnt 0x8
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
	s_xor_b32 s18, s28, -1
	s_mov_b32 s30, 1
	v_add_f32_e32 v130, v130, v13
	v_fmac_f32_e32 v9, v14, v51
	v_cvt_f32_i32_e32 v14, v16
	v_mul_f32_e32 v13, v90, v77
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s18
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
	s_or_b32 s18, s30, s21
	s_lshl_b32 s34, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[18:19], s[6:7]
	s_mov_b32 s35, s19
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[10:11], s[36:37]
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
	s_add_co_i32 s18, s18, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[18:19], s[6:7]
	s_add_co_i32 s18, s30, s20
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[18:19], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[10:11], s[34:35]
	s_mulk_i32 s18, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[76:77], null, 0x48, v188, s[34:35]
	v_add_co_u32 v72, s27, s34, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s27
	s_xor_b32 s27, s30, 1
	s_add_nc_u64 s[30:31], s[8:9], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s27, 6
	s_mov_b32 s37, s19
	v_add_co_u32 v76, vcc_lo, v76, v195
	v_add_co_u32 v74, s33, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v77, null, 0, v77, vcc_lo
	v_add_co_u32 v82, vcc_lo, v65, s18
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v78, s33, s30, v69
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	v_add_co_u32 v80, s30, s30, v71
	s_lshl_b32 s18, s27, 2
	v_add_co_ci_u32_e64 v79, null, s31, 0, s33
	v_add_co_ci_u32_e64 v81, null, s31, 0, s30
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s18
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
.LBB3_12:                               ; %.preheader510.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_add_nc_u32_e32 v84, 0, v190
	v_add_nc_u32_e32 v85, 0, v189
	s_xor_b32 s18, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s5, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v84 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v85 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v85 offset0:34 offset1:35
	s_cselect_b32 s27, s17, 0x3c00
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
	s_and_not1_b32 vcc_lo, exec_lo, s18
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
; %bb.13:                               ; %.preheader511.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v192, v197
	s_and_b32 s18, s29, exec_lo
	s_cselect_b32 s18, s5, 0x3400
	v_cndmask_b32_e64 v119, 0, v119, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s18, v191
	v_cvt_f32_f16_e64 v198, v198.l
	s_cselect_b32 s18, s17, 0x3c00
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v199, 0, v196, s2
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v201, s18, v191
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v186, v[118:119], v[112:113] offset1:8
	ds_store_2addr_stride64_b64 v187, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB3_9
.LBB3_14:                               ; %._crit_edge582.i
	v_mul_u32_u24_e32 v1, 0x500, v177
	v_lshlrev_b32_e32 v2, 2, v167
	s_ashr_i32 s17, s16, 31
	s_ashr_i32 s5, s4, 31
	s_ashr_i32 s15, s14, 31
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[4:5], s[16:17]
	v_add3_u32 v5, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[14:15], 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[12:13], s[0:1]
	v_lshrrev_b32_e32 v0, 4, v0
	v_mad_u32_u24 v1, 0x280, v168, v5
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[0:1], s[2:3]
	s_add_co_i32 s5, s14, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s9, s9, 0xffff
	ds_store_2addr_b32 v1, v178, v185 offset1:20
	ds_store_2addr_b32 v1, v183, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v181, v182 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v179, v180 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_cmp_gt_i32 s5, s4
	v_mul_lo_u32 v4, s4, v0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s16, 0x80
	v_lshl_add_u32 v1, v0, 2, 0
	v_or_b32_e32 v3, s16, v0
	v_or_b32_e32 v0, s14, v167
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s6
	s_mov_b32 s11, 0x31004000
	s_cselect_b32 s1, -1, 0
	v_mad_u32_u24 v2, 0x50, v167, v1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s3, s0, s1
	v_cmp_gt_i32_e64 s1, s4, v0
	v_cmp_gt_i32_e64 s0, s6, v3
	s_mov_b32 s10, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s2, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_32
; %bb.15:                               ; %.preheader505.i
	s_and_b32 s5, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s5
	s_cbranch_execz .LBB3_17
; %bb.16:
	v_mad_co_i64_i32 v[6:7], null, s4, v3, 0
	ds_load_b32 v10, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, vcc_lo, s12, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s13, v7, vcc_lo
	v_add_co_u32 v6, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off
.LBB3_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s6, v6
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_19
; %bb.18:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off
.LBB3_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_21
; %bb.20:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s2, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s2
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:128
.LBB3_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_23
; %bb.22:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:128
.LBB3_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_25
; %bb.24:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s2, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s2
	v_add_co_u32 v7, s2, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s2
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:256
.LBB3_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_27
; %bb.26:
	v_mad_co_i64_i32 v[7:8], null, s4, v6, 0
	ds_load_b32 v11, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s1, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s1
	v_add_co_u32 v7, s1, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:256
.LBB3_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s0, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_29
; %bb.28:
	v_mad_co_i64_i32 v[7:8], null, s4, v3, 0
	ds_load_b32 v11, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, s0, s12, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, s0
	v_add_co_u32 v7, s0, v1, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s0
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v11, off offset:384
.LBB3_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_31
; %bb.30:
	v_mad_co_i64_i32 v[6:7], null, s4, v6, 0
	ds_load_b32 v10, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[0:1]
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v1, vcc_lo, s12, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s13, v7, vcc_lo
	v_add_co_u32 v6, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v10, off offset:384
.LBB3_31:                               ; %Flow800
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB3_32:                               ; %Flow801
	v_add_lshl_u32 v4, v4, v167, 2
	v_mul_u32_u24_e32 v1, 0x280, v168
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_34
; %bb.33:                               ; %.preheader.i
	ds_load_2addr_stride64_b32 v[6:7], v2 offset1:5
	ds_load_2addr_stride64_b32 v[8:9], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[10:11], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[12:13], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 8
	s_movk_i32 s1, 0x80
	s_movk_i32 s2, 0x100
	s_movk_i32 s5, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s7, s0, 0x80
	s_add_co_i32 s14, s0, 0x100
	s_add_co_i32 s15, s0, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v6, v4, s[8:11], null offen
	buffer_store_b32 v7, v4, s[8:11], s0 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	buffer_store_b32 v9, v4, s[8:11], s7 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v10, v4, s[8:11], s2 offen
	buffer_store_b32 v11, v4, s[8:11], s14 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v12, v4, s[8:11], s5 offen
	buffer_store_b32 v13, v4, s[8:11], s15 offen
.LBB3_34:                               ; %.loopexit.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v5, v5, v1
	v_cndmask_b32_e64 v6, 0, 1, s3
	v_cmp_gt_i32_e64 s1, s4, v0
	v_or_b32_e32 v8, 16, v3
	s_mov_b32 s0, -1
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v176, v175 offset1:20
	ds_store_2addr_b32 v5, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v172, v171 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v170, v169 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_52
; %bb.35:                               ; %.preheader505.1.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_37
; %bb.36:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v7, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s0, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s0
	v_add_co_u32 v9, s0, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off
.LBB3_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v7, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_39
; %bb.38:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB3_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_41
; %bb.40:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB3_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_43
; %bb.42:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:128
.LBB3_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_45
; %bb.44:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s2, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s2
	v_add_co_u32 v9, s2, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s2
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB3_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_47
; %bb.46:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s1, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s1
	v_add_co_u32 v9, s1, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s1
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:256
.LBB3_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_49
; %bb.48:
	v_mad_co_i64_i32 v[9:10], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s12, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:384
.LBB3_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_51
; %bb.50:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s12, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v7, off offset:384
.LBB3_51:                               ; %Flow798
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_52:                               ; %Flow799
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s5, s4, 0x140
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_54
; %bb.53:                               ; %.preheader.1.i
	ds_load_2addr_stride64_b32 v[9:10], v2 offset1:5
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[15:16], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 6
	s_add_co_i32 s1, s5, 0x80
	s_add_co_i32 s2, s5, 0x100
	s_add_co_i32 s3, s5, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s7, s0, 0x80
	s_add_co_i32 s14, s0, 0x100
	s_add_co_i32 s15, s0, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s0 offen
	buffer_store_b32 v10, v4, s[8:11], s5 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s7 offen
	buffer_store_b32 v12, v4, s[8:11], s1 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s14 offen
	buffer_store_b32 v14, v4, s[8:11], s2 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v15, v4, s[8:11], s15 offen
	buffer_store_b32 v16, v4, s[8:11], s3 offen
.LBB3_54:                               ; %.loopexit.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_cmp_gt_i32_e64 s1, s4, v0
	v_or_b32_e32 v7, 32, v3
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v165, v166 offset1:20
	ds_store_2addr_b32 v5, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v159, v160 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_72
; %bb.55:                               ; %.preheader505.2.i
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_57
; %bb.56:
	v_mad_co_i64_i32 v[9:10], null, s4, v7, 0
	ds_load_b32 v13, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, s0, s12, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, s0
	v_add_co_u32 v9, s0, v1, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s0
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off
.LBB3_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v9, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v9
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_59
; %bb.58:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	ds_load_b32 v14, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
.LBB3_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_61
; %bb.60:
	v_mad_co_i64_i32 v[10:11], null, s4, v7, 0
	ds_load_b32 v14, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s2, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s2
	v_add_co_u32 v10, s2, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB3_61:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_63
; %bb.62:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	ds_load_b32 v14, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:128
.LBB3_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_65
; %bb.64:
	v_mad_co_i64_i32 v[10:11], null, s4, v7, 0
	ds_load_b32 v14, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s2, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s2
	v_add_co_u32 v10, s2, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s2
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB3_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_67
; %bb.66:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	ds_load_b32 v14, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s1, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s1
	v_add_co_u32 v10, s1, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s1
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:256
.LBB3_67:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_69
; %bb.68:
	v_mad_co_i64_i32 v[10:11], null, s4, v7, 0
	ds_load_b32 v14, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, vcc_lo, s12, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB3_69:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_71
; %bb.70:
	v_mad_co_i64_i32 v[9:10], null, s4, v9, 0
	ds_load_b32 v13, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v1, vcc_lo, s12, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s13, v10, vcc_lo
	v_add_co_u32 v9, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v13, off offset:384
.LBB3_71:                               ; %Flow796
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_72:                               ; %Flow797
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s3, s4, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_74
; %bb.73:                               ; %.preheader.2.i
	ds_load_2addr_stride64_b32 v[9:10], v2 offset1:5
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[15:16], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 7
	s_add_co_i32 s1, s3, 0x80
	s_add_co_i32 s2, s3, 0x100
	s_add_co_i32 s7, s3, 0x180
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s14, s0, 0x80
	s_add_co_i32 s15, s0, 0x100
	s_add_co_i32 s16, s0, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v9, v4, s[8:11], s0 offen
	buffer_store_b32 v10, v4, s[8:11], s3 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s14 offen
	buffer_store_b32 v12, v4, s[8:11], s1 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s15 offen
	buffer_store_b32 v14, v4, s[8:11], s2 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v15, v4, s[8:11], s16 offen
	buffer_store_b32 v16, v4, s[8:11], s7 offen
.LBB3_74:                               ; %.loopexit.2.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_or_b32_e32 v9, 48, v3
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v157, v158 offset1:20
	ds_store_2addr_b32 v5, v155, v156 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v153, v154 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v151, v152 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_92
; %bb.75:                               ; %.preheader505.3.i
	v_cmp_gt_i32_e64 s1, s4, v0
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_77
; %bb.76:
	v_mad_co_i64_i32 v[10:11], null, s4, v9, 0
	ds_load_b32 v14, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, s0, s12, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, s0
	v_add_co_u32 v10, s0, v1, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s0
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off
.LBB3_77:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v10
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_79
; %bb.78:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	ds_load_b32 v15, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off
.LBB3_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB3_81
; %bb.80:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	ds_load_b32 v15, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:128
.LBB3_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_83
; %bb.82:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	ds_load_b32 v15, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:128
.LBB3_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s7, s2
	s_cbranch_execz .LBB3_85
; %bb.84:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	ds_load_b32 v15, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:256
.LBB3_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s7
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_87
; %bb.86:
	v_mad_co_i64_i32 v[11:12], null, s4, v10, 0
	ds_load_b32 v15, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:256
.LBB3_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s7, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s7
	s_cbranch_execz .LBB3_89
; %bb.88:
	v_mad_co_i64_i32 v[11:12], null, s4, v9, 0
	ds_load_b32 v15, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:384
.LBB3_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_91
; %bb.90:
	v_mad_co_i64_i32 v[10:11], null, s4, v10, 0
	ds_load_b32 v14, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[0:1]
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v1, vcc_lo, s12, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s13, v11, vcc_lo
	v_add_co_u32 v10, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v14, off offset:384
.LBB3_91:                               ; %Flow794
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_92:                               ; %Flow795
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mul_i32 s14, s4, 0xc0
	s_mul_i32 s7, s4, 0x1c0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_94
; %bb.93:                               ; %.preheader.3.i
	ds_load_2addr_stride64_b32 v[10:11], v2 offset1:5
	ds_load_2addr_stride64_b32 v[12:13], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[14:15], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[16:17], v2 offset0:30 offset1:35
	s_add_co_i32 s0, s14, 0x80
	s_add_co_i32 s1, s7, 0x80
	s_add_co_i32 s2, s14, 0x100
	s_add_co_i32 s15, s7, 0x100
	s_add_co_i32 s16, s14, 0x180
	s_add_co_i32 s17, s7, 0x180
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v10, v4, s[8:11], s14 offen
	buffer_store_b32 v11, v4, s[8:11], s7 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v12, v4, s[8:11], s0 offen
	buffer_store_b32 v13, v4, s[8:11], s1 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v14, v4, s[8:11], s2 offen
	buffer_store_b32 v15, v4, s[8:11], s15 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v16, v4, s[8:11], s16 offen
	buffer_store_b32 v17, v4, s[8:11], s17 offen
.LBB3_94:                               ; %.loopexit.3.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	v_or_b32_e32 v10, 16, v0
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v149, v150 offset1:20
	ds_store_2addr_b32 v5, v147, v148 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v146, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v143, v144 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_112
; %bb.95:                               ; %.preheader505.1657.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v3
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_97
; %bb.96:
	v_mad_co_i64_i32 v[11:12], null, s4, v3, 0
	ds_load_b32 v15, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s0
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:64
.LBB3_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 64, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v11
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_99
; %bb.98:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB3_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB3_101
; %bb.100:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	ds_load_b32 v16, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_103
; %bb.102:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB3_105
; %bb.104:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	ds_load_b32 v16, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_107
; %bb.106:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_107:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s15, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s15
	s_cbranch_execz .LBB3_109
; %bb.108:
	v_mad_co_i64_i32 v[12:13], null, s4, v3, 0
	ds_load_b32 v16, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, vcc_lo, s12, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, vcc_lo
	v_add_co_u32 v12, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:448
.LBB3_109:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_111
; %bb.110:
	v_mad_co_i64_i32 v[11:12], null, s4, v11, 0
	ds_load_b32 v15, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:448
.LBB3_111:                              ; %Flow792
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_112:                              ; %Flow793
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_114
; %bb.113:                              ; %.preheader.1686.i
	ds_load_2addr_stride64_b32 v[11:12], v2 offset1:5
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[15:16], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[17:18], v2 offset0:30 offset1:35
	s_lshl_b32 s1, s4, 8
	s_mov_b32 s0, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s17, s1, 64
	s_movk_i32 s2, 0xc0
	s_movk_i32 s15, 0x140
	s_movk_i32 s16, 0x1c0
	s_or_b32 s18, s1, 0xc0
	s_add_co_i32 s19, s1, 0x140
	s_addk_co_i32 s1, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s0 offen
	buffer_store_b32 v12, v4, s[8:11], s17 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s2 offen
	buffer_store_b32 v14, v4, s[8:11], s18 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v15, v4, s[8:11], s15 offen
	buffer_store_b32 v16, v4, s[8:11], s19 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v17, v4, s[8:11], s16 offen
	buffer_store_b32 v18, v4, s[8:11], s1 offen
.LBB3_114:                              ; %.loopexit.1695.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v141, v142 offset1:20
	ds_store_2addr_b32 v5, v140, v139 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v137, v138 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v135, v136 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_132
; %bb.115:                              ; %.preheader505.1.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v8
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_117
; %bb.116:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	ds_load_b32 v15, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s0
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:64
.LBB3_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 0x50, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v11
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_119
; %bb.118:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:64
.LBB3_119:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB3_121
; %bb.120:
	v_mad_co_i64_i32 v[12:13], null, s4, v8, 0
	ds_load_b32 v16, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_121:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_123
; %bb.122:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:192
.LBB3_123:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s2
	s_cbranch_execz .LBB3_125
; %bb.124:
	v_mad_co_i64_i32 v[12:13], null, s4, v8, 0
	ds_load_b32 v16, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s2, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s2
	v_add_co_u32 v12, s2, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s2
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_125:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_127
; %bb.126:
	v_mad_co_i64_i32 v[12:13], null, s4, v11, 0
	ds_load_b32 v16, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, s1, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s1
	v_add_co_u32 v12, s1, v1, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s1
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v16, off offset:320
.LBB3_127:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s15, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s15
	s_cbranch_execz .LBB3_129
; %bb.128:
	v_mad_co_i64_i32 v[12:13], null, s4, v8, 0
	ds_load_b32 v8, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[0:1]
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v1, vcc_lo, s12, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, vcc_lo
	v_add_co_u32 v12, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[12:13], v8, off offset:448
.LBB3_129:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_131
; %bb.130:
	v_mad_co_i64_i32 v[11:12], null, s4, v11, 0
	ds_load_b32 v8, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v8, off offset:448
.LBB3_131:                              ; %Flow790
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_132:                              ; %Flow791
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_134
; %bb.133:                              ; %.preheader.1.1.i
	ds_load_2addr_stride64_b32 v[11:12], v2 offset1:5
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[15:16], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[17:18], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 6
	s_add_co_i32 s1, s5, 64
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s16, s0, 64
	s_add_co_i32 s2, s5, 0xc0
	s_add_co_i32 s15, s5, 0x140
	s_addk_co_i32 s5, 0x1c0
	s_add_co_i32 s17, s0, 0xc0
	s_add_co_i32 s18, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s16 offen
	buffer_store_b32 v12, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s17 offen
	buffer_store_b32 v14, v4, s[8:11], s2 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v15, v4, s[8:11], s18 offen
	buffer_store_b32 v16, v4, s[8:11], s15 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v17, v4, s[8:11], s0 offen
	buffer_store_b32 v18, v4, s[8:11], s5 offen
.LBB3_134:                              ; %.loopexit.1.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v133, v134 offset1:20
	ds_store_2addr_b32 v5, v131, v132 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v129, v130 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v127, v128 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_152
; %bb.135:                              ; %.preheader505.2.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v7
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_137
; %bb.136:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	ds_load_b32 v8, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s0, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s0
	v_add_co_u32 v11, s0, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s0
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v8, off offset:64
.LBB3_137:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v8, 0x60, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v8
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_139
; %bb.138:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	ds_load_b32 v15, v2 offset:1280
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:64
.LBB3_139:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_141
; %bb.140:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	ds_load_b32 v15, v2 offset:2560
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:192
.LBB3_141:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_143
; %bb.142:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	ds_load_b32 v15, v2 offset:3840
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:192
.LBB3_143:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s2
	s_cbranch_execz .LBB3_145
; %bb.144:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	ds_load_b32 v15, v2 offset:5120
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s2, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s2
	v_add_co_u32 v11, s2, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s2
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:320
.LBB3_145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_147
; %bb.146:
	v_mad_co_i64_i32 v[11:12], null, s4, v8, 0
	ds_load_b32 v15, v2 offset:6400
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, s1, s12, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, s1
	v_add_co_u32 v11, s1, v1, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s1
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v15, off offset:320
.LBB3_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v1
	s_and_b32 s5, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s5
	s_cbranch_execz .LBB3_149
; %bb.148:
	v_mad_co_i64_i32 v[11:12], null, s4, v7, 0
	ds_load_b32 v7, v2 offset:7680
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[0:1]
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v1, vcc_lo, s12, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s13, v12, vcc_lo
	v_add_co_u32 v11, vcc_lo, v1, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v7, off offset:448
.LBB3_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_151
; %bb.150:
	v_mad_co_i64_i32 v[7:8], null, s4, v8, 0
	ds_load_b32 v13, v2 offset:8960
	v_ashrrev_i32_e32 v1, 31, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[0:1]
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v1, vcc_lo, s12, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s13, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v13, off offset:448
.LBB3_151:                              ; %Flow788
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_152:                              ; %Flow789
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_154
; %bb.153:                              ; %.preheader.2.1.i
	ds_load_2addr_stride64_b32 v[7:8], v2 offset1:5
	ds_load_2addr_stride64_b32 v[11:12], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[13:14], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[15:16], v2 offset0:30 offset1:35
	s_lshl_b32 s0, s4, 7
	s_or_b32 s1, s3, 64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s15, s0, 64
	s_add_co_i32 s2, s3, 0xc0
	s_add_co_i32 s5, s3, 0x140
	s_addk_co_i32 s3, 0x1c0
	s_add_co_i32 s16, s0, 0xc0
	s_add_co_i32 s17, s0, 0x140
	s_addk_co_i32 s0, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s15 offen
	buffer_store_b32 v8, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v11, v4, s[8:11], s16 offen
	buffer_store_b32 v12, v4, s[8:11], s2 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v13, v4, s[8:11], s17 offen
	buffer_store_b32 v14, v4, s[8:11], s5 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v15, v4, s[8:11], s0 offen
	buffer_store_b32 v16, v4, s[8:11], s3 offen
.LBB3_154:                              ; %.loopexit.2.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v6
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v125, v126 offset1:20
	ds_store_2addr_b32 v5, v123, v124 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v121, v122 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v120, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_172
; %bb.155:                              ; %.preheader505.3.1.i
	v_cmp_gt_i32_e64 s1, s4, v10
	v_cmp_gt_i32_e32 vcc_lo, s6, v9
	v_ashrrev_i32_e32 v1, 31, v0
	s_and_b32 s0, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB3_157
; %bb.156:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	ds_load_b32 v10, v2
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s0, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s0, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s0
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:64
.LBB3_157:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v3, 0x70, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s6, v3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_159
; %bb.158:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	ds_load_b32 v10, v2 offset:1280
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:64
.LBB3_159:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_161
; %bb.160:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	ds_load_b32 v10, v2 offset:2560
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s2, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s2, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s2
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:192
.LBB3_161:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_163
; %bb.162:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	ds_load_b32 v10, v2 offset:3840
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:192
.LBB3_163:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s2, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_165
; %bb.164:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	ds_load_b32 v10, v2 offset:5120
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s2, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s2, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s2
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:320
.LBB3_165:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_167
; %bb.166:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	ds_load_b32 v10, v2 offset:6400
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, s12, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, s1, v5, v7
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, s1
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v10, off offset:320
.LBB3_167:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s1, s4, v5
	s_and_b32 s3, s1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_169
; %bb.168:
	v_mad_co_i64_i32 v[5:6], null, s4, v9, 0
	ds_load_b32 v9, v2 offset:7680
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s12, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v5, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[5:6], v9, off offset:448
.LBB3_169:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_171
; %bb.170:
	v_mad_co_i64_i32 v[5:6], null, s4, v3, 0
	ds_load_b32 v3, v2 offset:8960
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, s12, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s13, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v5, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, v6, v1, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v3, off offset:448
.LBB3_171:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, 0
.LBB3_172:                              ; %Flow787
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_174
; %bb.173:                              ; %.preheader.3.1.i
	ds_load_2addr_stride64_b32 v[0:1], v2 offset1:5
	ds_load_2addr_stride64_b32 v[5:6], v2 offset0:10 offset1:15
	ds_load_2addr_stride64_b32 v[7:8], v2 offset0:20 offset1:25
	ds_load_2addr_stride64_b32 v[2:3], v2 offset0:30 offset1:35
	s_add_co_i32 s0, s14, 64
	s_add_co_i32 s1, s7, 64
	s_add_co_i32 s2, s14, 0xc0
	s_add_co_i32 s3, s7, 0xc0
	s_add_co_i32 s4, s14, 0x140
	s_add_co_i32 s5, s7, 0x140
	s_addk_co_i32 s14, 0x1c0
	s_addk_co_i32 s7, 0x1c0
	s_wait_dscnt 0x3
	s_clause 0x1
	buffer_store_b32 v0, v4, s[8:11], s0 offen
	buffer_store_b32 v1, v4, s[8:11], s1 offen
	s_wait_dscnt 0x2
	s_clause 0x1
	buffer_store_b32 v5, v4, s[8:11], s2 offen
	buffer_store_b32 v6, v4, s[8:11], s3 offen
	s_wait_dscnt 0x1
	s_clause 0x1
	buffer_store_b32 v7, v4, s[8:11], s4 offen
	buffer_store_b32 v8, v4, s[8:11], s5 offen
	s_wait_dscnt 0x0
	s_clause 0x1
	buffer_store_b32 v2, v4, s[8:11], s14 offen
	buffer_store_b32 v3, v4, s[8:11], s7 offen
.LBB3_174:                              ; %.loopexit.3.1.i
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_175:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
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
; codeLenInByte = 15900
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
	.type	__hip_cuid_ebbfd6bd3f8a1de5,@object ; @__hip_cuid_ebbfd6bd3f8a1de5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ebbfd6bd3f8a1de5
__hip_cuid_ebbfd6bd3f8a1de5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ebbfd6bd3f8a1de5, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ebbfd6bd3f8a1de5
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
