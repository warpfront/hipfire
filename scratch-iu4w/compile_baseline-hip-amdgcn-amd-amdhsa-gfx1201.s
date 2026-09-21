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
.LBB0_16:                               ; %_Z26quantize_block_i4_128_waveILb0EEvPKfP12block_i4_128i.exit
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
	s_lshl_b32 s22, ttmp9, 7
	s_lshl_b32 s24, ttmp7, 7
	v_lshrrev_b32_e32 v100, 5, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s22, s12
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s24, s14
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s11, s0, s1
	s_cmp_eq_u32 s15, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB1_15
; %bb.1:
	s_mov_b32 s28, 0
	s_and_b32 vcc_lo, exec_lo, s11
	s_cbranch_vccz .LBB1_16
; %bb.2:                                ; %.preheader530.i
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
	v_or_b32_e32 v5, s24, v1
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
.LBB1_5:                                ; %.preheader525.loopexit.i
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
.LBB1_6:                                ; %Flow1679
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v102, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s24, v9
	v_dual_mov_b32 v101, 0 :: v_dual_add_nc_u32 v2, s22, v9
	v_dual_mov_b32 v132, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v106, 0 :: v_dual_add_nc_u32 v3, 64, v2
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v154, 0 :: v_dual_and_b32 v13, 16, v13
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
	v_or_b32_e32 v14, 8, v100
	v_cmp_gt_i32_e64 s0, s14, v10
	v_mov_b32_e32 v128, 0
	v_and_or_b32 v13, v100, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_mov_b32_e32 v160, 0
	v_dual_mov_b32 v103, 0 :: v_dual_and_b32 v140, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v123, 0
	v_bfe_u32 v165, v0, 4, 1
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v136, 0
	v_add_nc_u32_e32 v166, 0, v13
	v_add_nc_u32_e32 v167, 0, v9
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v164, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v157, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v8, 0x88888888, v8
	v_xor_b32_e32 v7, 0x88888888, v7
	ds_store_2addr_stride64_b64 v166, v[1:2], v[3:4] offset1:8
	ds_store_2addr_stride64_b64 v167, v[5:6], v[7:8] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_17
; %bb.7:                                ; %.preheader524.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v157, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v176, 0 :: v_dual_add_nc_u32 v5, s22, v1
	v_dual_mov_b32 v177, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v164, 0 :: v_dual_lshlrev_b32 v7, 6, v165
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v163, 0 :: v_dual_lshlrev_b32 v8, 3, v140
	v_dual_mov_b32 v161, 0 :: v_dual_lshlrev_b32 v10, 8, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v159, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v162, 0 :: v_dual_add_nc_u32 v11, s24, v1
	v_dual_mov_b32 v160, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v131, 0 :: v_dual_lshlrev_b32 v12, 2, v6
	v_dual_mov_b32 v158, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v168, v4, 11, v2
	v_lshl_or_b32 v169, v3, 10, v2
	v_min_i32_e32 v170, s4, v11
	v_cmp_gt_i32_e64 s2, s14, v11
	v_add3_u32 v171, 0, v1, v12
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v129, 0 :: v_dual_lshlrev_b32 v172, 4, v6
	v_add3_u32 v173, 0, v10, v7
	v_add3_u32 v174, 0, v13, v8
	v_dual_mov_b32 v130, 0 :: v_dual_lshlrev_b32 v175, 2, v6
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v101, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s29, 0x1000
	s_movk_i32 s23, 0x3000
	s_movk_i32 s25, 0x3800
	s_mov_b32 s30, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB1_9
.LBB1_8:                                ;   in Loop: Header=BB1_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s27
	s_mov_b32 s6, s26
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_9:                                ; %.preheader524.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_11 Depth 2
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
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=2
	s_wait_loadcnt_dscnt 0x205
	v_dual_mul_f32 v84, v98, v82 :: v_dual_mul_f32 v85, v99, v82
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_dual_fmac_f32 v157, v84, v57 :: v_dual_mul_f32 v84, v97, v82
	v_mul_f32_e32 v57, v96, v82
	v_fmac_f32_e32 v164, v85, v58
	v_mul_f32_e32 v58, v94, v82
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v163, v84, v60 :: v_dual_fmac_f32 v162, v57, v59
	v_mul_f32_e32 v57, v95, v82
	v_cvt_f32_i32_e32 v59, v62
	v_fmac_f32_e32 v160, v58, v61
	v_mul_f32_e32 v58, v88, v82
	v_cvt_f32_i32_e32 v61, v63
	v_mul_f32_e32 v60, v89, v82
	v_fmac_f32_e32 v161, v57, v59
	v_mul_f32_e32 v57, v98, v83
	v_cvt_f32_i32_e32 v62, v64
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v53, v53
	v_fmac_f32_e32 v155, v57, v49
	v_dual_mul_f32 v49, v96, v83 :: v_dual_fmac_f32 v158, v58, v61
	v_dual_fmac_f32 v159, v60, v62 :: v_dual_mul_f32 v58, v99, v83
	v_mul_f32_e32 v57, v97, v83
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	v_fmac_f32_e32 v156, v58, v50
	v_mul_f32_e32 v50, v94, v83
	v_fmac_f32_e32 v153, v49, v51
	v_mul_f32_e32 v49, v95, v83
	v_cvt_f32_i32_e32 v51, v54
	v_cvt_f32_i32_e32 v54, v56
	v_fmac_f32_e32 v152, v50, v53
	v_mul_f32_e32 v50, v88, v83
	v_cvt_f32_i32_e32 v53, v55
	v_fmac_f32_e32 v151, v49, v51
	s_wait_dscnt 0x4
	v_mul_f32_e32 v49, v98, v72
	v_fmac_f32_e32 v154, v57, v52
	v_dual_mul_f32 v52, v89, v83 :: v_dual_fmac_f32 v149, v50, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v50, v99, v72 :: v_dual_fmac_f32 v147, v49, v41
	v_mul_f32_e32 v41, v96, v72
	v_cvt_f32_i32_e32 v43, v43
	v_dual_mul_f32 v49, v97, v72 :: v_dual_fmac_f32 v148, v50, v42
	v_cvt_f32_i32_e32 v44, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v42, v94, v72 :: v_dual_fmac_f32 v145, v41, v43
	v_cvt_f32_i32_e32 v45, v45
	v_mul_f32_e32 v41, v95, v72
	v_cvt_f32_i32_e32 v43, v46
	v_fmac_f32_e32 v146, v49, v44
	v_dual_mul_f32 v44, v89, v72 :: v_dual_fmac_f32 v143, v42, v45
	v_mul_f32_e32 v42, v88, v72
	v_cvt_f32_i32_e32 v45, v47
	v_dual_fmac_f32 v144, v41, v43 :: v_dual_mul_f32 v41, v98, v73
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v46, v48
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v141, v42, v45
	v_mul_f32_e32 v42, v99, v73
	v_cvt_f32_i32_e32 v34, v34
	v_fmac_f32_e32 v138, v41, v33
	v_mul_f32_e32 v33, v96, v73
	v_cvt_f32_i32_e32 v35, v35
	v_dual_fmac_f32 v142, v44, v46 :: v_dual_mul_f32 v41, v97, v73
	v_cvt_f32_i32_e32 v36, v36
	v_fmac_f32_e32 v139, v42, v34
	v_mul_f32_e32 v34, v94, v73
	v_cvt_f32_i32_e32 v37, v37
	v_dual_fmac_f32 v136, v33, v35 :: v_dual_mul_f32 v33, v95, v73
	v_cvt_f32_i32_e32 v35, v38
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v137, v41, v36 :: v_dual_fmac_f32 v134, v34, v37
	v_mul_f32_e32 v34, v88, v73
	v_cvt_f32_i32_e32 v37, v39
	v_fmac_f32_e32 v135, v33, v35
	s_wait_dscnt 0x3
	v_mul_f32_e32 v33, v82, v80
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v36, v89, v73
	v_cvt_f32_i32_e32 v38, v40
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v130, v33, v25 :: v_dual_mul_f32 v25, v82, v78
	v_dual_fmac_f32 v132, v34, v37 :: v_dual_fmac_f32 v133, v36, v38
	v_dual_mul_f32 v34, v82, v81 :: v_dual_mul_f32 v33, v82, v79
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v34, v26
	s_wait_dscnt 0x1
	v_mul_f32_e32 v26, v82, v76
	v_dual_fmac_f32 v128, v25, v27 :: v_dual_mul_f32 v25, v82, v77
	v_cvt_f32_i32_e32 v27, v30
	v_fmac_f32_e32 v129, v33, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v127, v26, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v82, v74
	v_mul_f32_e32 v28, v82, v75
	v_cvt_f32_i32_e32 v29, v31
	v_cvt_f32_i32_e32 v30, v32
	v_dual_fmac_f32 v126, v25, v27 :: v_dual_mul_f32 v25, v83, v80
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v26, v29
	v_mul_f32_e32 v26, v83, v81
	v_cvt_f32_i32_e32 v19, v19
	v_dual_fmac_f32 v122, v25, v17 :: v_dual_mul_f32 v17, v83, v78
	v_mul_f32_e32 v25, v83, v79
	v_cvt_f32_i32_e32 v20, v20
	v_dual_fmac_f32 v123, v26, v18 :: v_dual_mul_f32 v18, v83, v76
	v_cvt_f32_i32_e32 v21, v21
	v_dual_fmac_f32 v120, v17, v19 :: v_dual_mul_f32 v17, v83, v77
	v_cvt_f32_i32_e32 v19, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v121, v25, v20 :: v_dual_fmac_f32 v118, v18, v21
	v_mul_f32_e32 v18, v83, v74
	v_cvt_f32_i32_e32 v21, v23
	v_fmac_f32_e32 v119, v17, v19
	v_mul_f32_e32 v17, v72, v80
	v_cvt_f32_i32_e32 v9, v9
	v_mul_f32_e32 v20, v83, v75
	v_cvt_f32_i32_e32 v22, v24
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_dual_fmac_f32 v114, v17, v9 :: v_dual_mul_f32 v9, v72, v78
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v116, v18, v21 :: v_dual_fmac_f32 v117, v20, v22
	v_dual_mul_f32 v18, v72, v81 :: v_dual_mul_f32 v17, v72, v79
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_dual_fmac_f32 v112, v9, v11 :: v_dual_fmac_f32 v115, v18, v10
	v_dual_mul_f32 v10, v72, v76 :: v_dual_mul_f32 v9, v72, v77
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v113, v17, v12
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v11, v14
	v_fmac_f32_e32 v110, v10, v13
	v_mul_f32_e32 v10, v72, v74
	v_cvt_f32_i32_e32 v13, v15
	v_dual_mul_f32 v15, v73, v80 :: v_dual_mul_f32 v12, v72, v75
	v_cvt_f32_i32_e32 v14, v16
	v_cvt_f32_i32_e32 v2, v2
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v106, v15, v1 :: v_dual_mul_f32 v1, v73, v78
	v_dual_mul_f32 v16, v73, v81 :: v_dual_fmac_f32 v109, v12, v14
	s_barrier_signal -1
	v_dual_fmac_f32 v108, v10, v13 :: v_dual_fmac_f32 v111, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v107, v16, v2
	v_cvt_f32_i32_e32 v2, v3
	v_mul_f32_e32 v3, v73, v79
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v104, v1, v2
	v_dual_mul_f32 v1, v73, v76 :: v_dual_mul_f32 v2, v73, v77
	v_dual_mul_f32 v9, v73, v74 :: v_dual_mul_f32 v10, v73, v75
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v150, v52, v54
	v_fmac_f32_e32 v125, v28, v30
	v_dual_fmac_f32 v105, v3, v4 :: v_dual_fmac_f32 v102, v1, v5
	v_fmac_f32_e32 v103, v2, v6
	v_fmac_f32_e32 v101, v9, v7
	v_fmac_f32_e32 v67, v10, v8
	s_xor_b32 s4, s31, -1
	s_mov_b32 s34, 1
	s_mov_b32 s31, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s33, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_8
.LBB1_11:                               ;   Parent Loop BB1_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s34, 6
	s_mov_b32 s37, s5
	s_or_b32 s4, s34, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[8:9], s[36:37]
	s_mul_u64 s[38:39], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s35, s36, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s35
	v_add_co_u32 v3, s35, s36, v71
	s_mul_u64 s[38:39], s[38:39], 0x48
	v_add_co_ci_u32_e64 v4, null, s37, 0, s35
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[38:39], s[18:19], s[38:39]
	s_clause 0x1
	global_load_b64 v[7:8], v[3:4], off offset:40
	global_load_b64 v[80:81], v[1:2], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s35, s38, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s39, 0, s35
	v_add_co_u32 v1, s35, s38, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s39, 0, s35
	s_clause 0x1
	global_load_b64 v[92:93], v[5:6], off offset:40
	global_load_b64 v[84:85], v[1:2], off offset:40
	v_add_nc_u32_e32 v82, s29, v169
	v_add_nc_u32_e32 v83, s30, v168
	ds_load_2addr_stride64_b64 v[72:75], v82 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v83 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v83 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	v_xor_b32_e32 v87, 0x88888888, v8
	v_xor_b32_e32 v86, 0x88888888, v7
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v91, 0x88888888, v81
	v_xor_b32_e32 v90, 0x88888888, v80
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v82 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v83 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v83 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v93, s0
	v_cndmask_b32_e64 v72, 0, v92, s0
	s_and_b32 s29, s33, s27
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s29
	ds_store_2addr_stride64_b64 v166, v[90:91], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v85, s1
	v_cndmask_b32_e64 v72, 0, v84, s1
	ds_store_2addr_stride64_b64 v167, v[86:87], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=2
	s_add_co_i32 s4, s4, 1
	s_xor_b32 s30, s34, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_add_co_i32 s4, s34, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_mul_u64 s[34:35], s[4:5], 0x88
	s_mov_b32 s39, s5
	s_lshl_b32 s38, s30, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_add_nc_u64 s[34:35], s[16:17], s[34:35]
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[34:35], s[38:39]
	v_mad_co_i64_i32 v[80:81], null, 0x48, v170, s[36:37]
	v_add_co_u32 v82, vcc_lo, v65, s4
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s38, s34, v69
	v_add_co_u32 v76, s34, s34, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s35, 0, s38
	s_lshl_b32 s4, s30, 2
	v_add_co_u32 v72, s40, s36, v68
	v_add_co_ci_u32_e64 v77, null, s35, 0, s34
	v_add_co_u32 v78, s34, s36, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s37, 0, s40
	v_add_co_ci_u32_e64 v79, null, s37, 0, s34
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	v_add_co_u32 v80, vcc_lo, v80, v175
	s_clause 0x1
	global_load_b64 v[74:75], v[74:75], off offset:8
	global_load_b64 v[76:77], v[76:77], off offset:8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[92:93], v[72:73], off offset:8
	global_load_b64 v[84:85], v[78:79], off offset:8
	global_load_b32 v176, v[82:83], off
	global_load_b32 v177, v[80:81], off
	s_wait_loadcnt 0x5
	v_xor_b32_e32 v91, 0x88888888, v75
	s_wait_loadcnt 0x4
	v_xor_b32_e32 v87, 0x88888888, v77
	v_xor_b32_e32 v86, 0x88888888, v76
	v_xor_b32_e32 v90, 0x88888888, v74
.LBB1_13:                               ; %.preheader522.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	v_add_nc_u32_e32 v88, 0, v169
	v_add_nc_u32_e32 v89, 0, v168
	s_xor_b32 s4, s29, -1
	s_and_b32 s29, s31, exec_lo
	s_cselect_b32 s29, s23, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v88 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v89 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v89 offset0:34 offset1:35
	s_cselect_b32 s30, s25, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v88
	v_add_nc_u32_e32 v80, 0x100, v89
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s30, v173
	v_add_nc_u32_e32 v72, s29, v174
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_movk_i32 s29, 0x2000
	s_movk_i32 s30, 0x4000
	ds_load_2addr_b32 v[98:99], v74 offset1:2
	ds_load_2addr_b32 v[96:97], v74 offset0:4 offset1:6
	ds_load_2addr_b32 v[94:95], v74 offset0:8 offset1:10
	ds_load_2addr_b32 v[88:89], v74 offset0:12 offset1:14
	ds_load_2addr_b32 v[82:83], v72 offset1:32
	ds_load_2addr_b32 v[72:73], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[80:81], v74 offset0:32 offset1:34
	ds_load_2addr_b32 v[78:79], v74 offset0:36 offset1:38
	ds_load_2addr_b32 v[76:77], v74 offset0:40 offset1:42
	ds_load_2addr_b32 v[74:75], v74 offset0:44 offset1:46
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
; %bb.14:                               ; %.preheader523.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v172, v176
	s_and_b32 s4, s33, exec_lo
	s_cselect_b32 s4, s23, 0x3400
	v_cndmask_b32_e64 v93, 0, v93, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v180, s4, v171
	v_cvt_f32_f16_e64 v178, v178.l
	s_cselect_b32 s4, s25, 0x3c00
	v_cndmask_b32_e64 v92, 0, v92, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v179, 0, v177, s2
	v_cndmask_b32_e64 v85, 0, v85, s1
	v_cndmask_b32_e64 v84, 0, v84, s1
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v181, s4, v171
	s_mov_b32 s30, 0
	s_movk_i32 s29, 0x1000
	ds_store_2addr_stride64_b64 v166, v[92:93], v[90:91] offset1:8
	ds_store_2addr_stride64_b64 v167, v[84:85], v[86:87] offset1:8
	ds_store_b32 v180, v179
	ds_store_b32 v181, v178
	s_branch .LBB1_10
.LBB1_15:
	s_mov_b32 s28, -1
.LBB1_16:                               ; %Flow1688
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s28
	s_cbranch_vccnz .LBB1_111
	s_branch .LBB1_219
.LBB1_17:                               ; %._crit_edge590.i
	v_mul_u32_u24_e32 v1, 0x500, v100
	v_lshlrev_b32_e32 v2, 2, v140
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
	v_mul_u32_u24_e32 v4, 0x50, v140
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v6
	v_mad_u32_u24 v3, 0x280, v165, v8
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_mov_b32 s26, s12
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_108
; %bb.18:                               ; %.preheader519.i
	ds_store_2addr_b32 v3, v157, v164 offset1:20
	ds_store_2addr_b32 v3, v162, v163 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v160, v161 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v158, v159 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v1, s22, v140
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
	s_cbranch_execz .LBB1_20
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
.LBB1_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v10, 64, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v10
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_22
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
.LBB1_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v11
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_24
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
.LBB1_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_26
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
.LBB1_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v11
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_28
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
.LBB1_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_30
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
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v11, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v11
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_32
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
.LBB1_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v11, 0x280, v165
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_34
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
.LBB1_34:                               ; %.preheader517.1.i
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
	ds_store_2addr_b32 v8, v155, v156 offset1:20
	ds_store_2addr_b32 v8, v153, v154 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v152, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v149, v150 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_36
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
.LBB1_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v12, 0x50, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v12
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_222
; %bb.37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_223
.LBB1_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_224
.LBB1_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_225
.LBB1_40:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_226
.LBB1_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_227
.LBB1_42:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_44
.LBB1_43:
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
.LBB1_44:                               ; %.preheader517.2.i
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
	ds_store_2addr_b32 v8, v147, v148 offset1:20
	ds_store_2addr_b32 v8, v145, v146 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v143, v144 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v141, v142 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_46
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
.LBB1_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v14, 0x60, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v14
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_228
; %bb.47:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_229
.LBB1_48:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_230
.LBB1_49:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_231
.LBB1_50:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_232
.LBB1_51:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_233
.LBB1_52:
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_54
.LBB1_53:
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
.LBB1_54:                               ; %.preheader517.3.i
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
	ds_store_2addr_b32 v8, v138, v139 offset1:20
	ds_store_2addr_b32 v8, v136, v137 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v134, v135 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v132, v133 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s15, s6
	s_cbranch_execz .LBB1_56
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
.LBB1_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v16, 0x70, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v16
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_234
; %bb.57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execnz .LBB1_235
.LBB1_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_236
.LBB1_59:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_237
.LBB1_60:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_238
.LBB1_61:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_239
.LBB1_62:
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_64
.LBB1_63:
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
.LBB1_64:                               ; %.preheader518.1.i
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
	ds_store_2addr_b32 v8, v130, v131 offset1:20
	ds_store_2addr_b32 v8, v128, v129 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v127, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v124, v125 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_66
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
.LBB1_66:
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_68
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
.LBB1_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v17, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v17
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_70
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
.LBB1_70:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_72
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
.LBB1_72:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v17, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v17
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_74
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
.LBB1_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_76
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
.LBB1_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v17, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v17
	s_and_b32 s23, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s23
	s_cbranch_execz .LBB1_78
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
.LBB1_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s15, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_80
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
.LBB1_80:                               ; %.preheader517.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s15, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v122, v123 offset1:20
	ds_store_2addr_b32 v8, v120, v121 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v118, v119 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v116, v117 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_240
; %bb.81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_241
.LBB1_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_242
.LBB1_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_243
.LBB1_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_244
.LBB1_85:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execnz .LBB1_245
.LBB1_86:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_246
.LBB1_87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_89
.LBB1_88:
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
.LBB1_89:                               ; %.preheader517.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v114, v115 offset1:20
	ds_store_2addr_b32 v8, v112, v113 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v110, v111 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v108, v109 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_247
; %bb.90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_248
.LBB1_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_249
.LBB1_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_250
.LBB1_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_251
.LBB1_94:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_252
.LBB1_95:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_253
.LBB1_96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_98
.LBB1_97:
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
.LBB1_98:                               ; %.preheader517.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v8, v106, v107 offset1:20
	ds_store_2addr_b32 v8, v104, v105 offset0:40 offset1:60
	ds_store_2addr_b32 v8, v102, v103 offset0:80 offset1:100
	ds_store_2addr_b32 v8, v101, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_254
; %bb.99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_255
.LBB1_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_256
.LBB1_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_257
.LBB1_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_258
.LBB1_103:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_259
.LBB1_104:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_260
.LBB1_105:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_107
.LBB1_106:
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
.LBB1_107:                              ; %.loopexit.loopexit623.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB1_108:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_110
; %bb.109:                              ; %.preheader516.i
	ds_store_2addr_b32 v3, v157, v164 offset1:20
	ds_store_2addr_b32 v3, v162, v163 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v160, v161 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v158, v159 offset0:120 offset1:140
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
	v_add_lshl_u32 v6, v1, v140, 2
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
	ds_store_2addr_b32 v3, v155, v156 offset1:20
	ds_store_2addr_b32 v3, v153, v154 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v152, v151 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v149, v150 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v147, v148 offset1:20
	ds_store_2addr_b32 v3, v145, v146 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v143, v144 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v141, v142 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v138, v139 offset1:20
	ds_store_2addr_b32 v3, v136, v137 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v134, v135 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v132, v133 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v130, v131 offset1:20
	ds_store_2addr_b32 v3, v128, v129 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v127, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v124, v125 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v122, v123 offset1:20
	ds_store_2addr_b32 v3, v120, v121 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v118, v119 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v116, v117 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v114, v115 offset1:20
	ds_store_2addr_b32 v3, v112, v113 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v110, v111 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v108, v109 offset0:120 offset1:140
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
	ds_store_2addr_b32 v3, v106, v107 offset1:20
	ds_store_2addr_b32 v3, v104, v105 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v102, v103 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v101, v67 offset0:120 offset1:140
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
.LBB1_110:                              ; %Flow1674
	s_mov_b32 s0, -1
	s_and_b32 vcc_lo, exec_lo, s28
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_219
.LBB1_111:
	s_and_b32 vcc_lo, exec_lo, s11
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_219
; %bb.112:                              ; %.preheader525.i
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
	s_cbranch_execz .LBB1_116
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
	s_cbranch_execz .LBB1_115
; %bb.114:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_115:                              ; %.preheader520.loopexit.i
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
.LBB1_116:                              ; %Flow1687
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v0
	v_dual_mov_b32 v102, 0 :: v_dual_and_b32 v1, 3, v0
	s_add_co_i32 s4, s14, -1
	v_dual_mov_b32 v136, 0 :: v_dual_lshlrev_b32 v13, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, s24, v9
	v_dual_mov_b32 v101, 0 :: v_dual_add_nc_u32 v2, s22, v9
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v11, 64, v10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v106, 0 :: v_dual_add_nc_u32 v3, 64, v2
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v4, s4, v10
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s4, v11
	v_min_i32_e32 v3, s2, v3
	v_dual_mov_b32 v154, 0 :: v_dual_and_b32 v13, 16, v13
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
	v_or_b32_e32 v14, 8, v100
	v_cmp_gt_i32_e64 s0, s14, v10
	v_mov_b32_e32 v128, 0
	v_and_or_b32 v13, v100, 6, v12
	v_lshlrev_b32_e32 v9, 3, v9
	v_and_or_b32 v12, v14, 14, v12
	v_cmp_gt_i32_e64 s1, s14, v11
	v_mov_b32_e32 v162, 0
	v_dual_mov_b32 v103, 0 :: v_dual_and_b32 v124, 15, v0
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_mov_b32_e32 v123, 0
	v_bfe_u32 v141, v0, 4, 1
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v138, 0
	v_add_nc_u32_e32 v166, 0, v13
	v_add_nc_u32_e32 v167, 0, v9
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v158, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v165, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v8, 0x88888888, v8
	v_xor_b32_e32 v7, 0x88888888, v7
	ds_store_2addr_stride64_b64 v166, v[1:2], v[3:4] offset1:8
	ds_store_2addr_stride64_b64 v167, v[5:6], v[7:8] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_125
; %bb.117:                              ; %.preheader519.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v0
	v_dual_mov_b32 v177, 0 :: v_dual_and_b32 v2, 31, v0
	v_lshrrev_b32_e32 v3, 6, v0
	v_bfe_u32 v4, v0, 5, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v158, 0 :: v_dual_add_nc_u32 v5, s22, v1
	v_dual_mov_b32 v165, 0 :: v_dual_and_b32 v6, 1, v0
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v7, 6, v141
	v_min_i32_e32 v9, s2, v5
	v_dual_mov_b32 v163, 0 :: v_dual_lshlrev_b32 v8, 3, v124
	v_dual_mov_b32 v161, 0 :: v_dual_lshlrev_b32 v10, 8, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[65:66], null, s3, v9, s[16:17]
	v_ashrrev_i32_e32 v9, 31, v9
	v_dual_mov_b32 v159, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v164, 0 :: v_dual_add_nc_u32 v11, s24, v1
	v_dual_mov_b32 v162, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v131, 0 :: v_dual_lshlrev_b32 v12, 2, v6
	v_dual_mov_b32 v160, 0 :: v_dual_lshlrev_b32 v13, 9, v4
	v_mad_co_u64_u32 v[66:67], null, s3, v9, v[66:67]
	v_lshl_or_b32 v168, v4, 11, v2
	v_lshl_or_b32 v169, v3, 10, v2
	v_min_i32_e32 v170, s4, v11
	v_cmp_gt_i32_e64 s2, s14, v11
	v_add3_u32 v171, 0, v1, v12
	v_cmp_gt_i32_e64 s3, s12, v5
	v_dual_mov_b32 v129, 0 :: v_dual_lshlrev_b32 v172, 4, v6
	v_add3_u32 v173, 0, v10, v7
	v_add3_u32 v174, 0, v13, v8
	v_dual_mov_b32 v132, 0 :: v_dual_lshlrev_b32 v175, 2, v6
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v101, 0
	v_mov_b32_e32 v67, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s11, 0x3000
	s_movk_i32 s13, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_branch .LBB1_119
.LBB1_118:                              ;   in Loop: Header=BB1_119 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s6, s23
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_125
.LBB1_119:                              ; %.preheader519.i35
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_121 Depth 2
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
	s_branch .LBB1_121
.LBB1_120:                              ;   in Loop: Header=BB1_121 Depth=2
	s_wait_loadcnt_dscnt 0x205
	v_dual_mul_f32 v84, v98, v82 :: v_dual_mul_f32 v85, v99, v82
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v158, v84, v57 :: v_dual_fmac_f32 v165, v85, v58
	v_dual_mul_f32 v57, v96, v82 :: v_dual_mul_f32 v84, v97, v82
	v_mul_f32_e32 v58, v94, v82
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_dual_fmac_f32 v163, v57, v59 :: v_dual_fmac_f32 v164, v84, v60
	v_mul_f32_e32 v57, v95, v82
	v_cvt_f32_i32_e32 v59, v62
	v_dual_fmac_f32 v161, v58, v61 :: v_dual_mul_f32 v58, v88, v82
	v_mul_f32_e32 v60, v89, v82
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_fmac_f32_e32 v162, v57, v59
	v_mul_f32_e32 v57, v98, v83
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v159, v58, v61 :: v_dual_fmac_f32 v160, v60, v62
	v_mul_f32_e32 v58, v99, v83
	v_dual_fmac_f32 v156, v57, v49 :: v_dual_mul_f32 v49, v96, v83
	v_mul_f32_e32 v57, v97, v83
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v157, v58, v50
	v_mul_f32_e32 v50, v94, v83
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v41, v41
	v_fmac_f32_e32 v155, v57, v52
	v_fmac_f32_e32 v154, v49, v51
	v_mul_f32_e32 v49, v95, v83
	v_cvt_f32_i32_e32 v51, v54
	v_dual_fmac_f32 v153, v50, v53 :: v_dual_mul_f32 v50, v88, v83
	v_mul_f32_e32 v52, v89, v83
	v_cvt_f32_i32_e32 v53, v55
	v_cvt_f32_i32_e32 v54, v56
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v152, v49, v51 :: v_dual_mul_f32 v49, v98, v72
	v_cvt_f32_i32_e32 v42, v42
	v_fmac_f32_e32 v150, v50, v53
	v_mul_f32_e32 v50, v99, v72
	v_cvt_f32_i32_e32 v43, v43
	v_dual_fmac_f32 v148, v49, v41 :: v_dual_mul_f32 v41, v96, v72
	v_mul_f32_e32 v49, v97, v72
	v_cvt_f32_i32_e32 v44, v44
	v_fmac_f32_e32 v149, v50, v42
	v_mul_f32_e32 v42, v94, v72
	v_cvt_f32_i32_e32 v45, v45
	v_dual_fmac_f32 v146, v41, v43 :: v_dual_mul_f32 v41, v95, v72
	v_cvt_f32_i32_e32 v43, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v147, v49, v44 :: v_dual_fmac_f32 v144, v42, v45
	v_mul_f32_e32 v42, v88, v72
	v_cvt_f32_i32_e32 v45, v47
	v_fmac_f32_e32 v145, v41, v43
	v_mul_f32_e32 v41, v98, v73
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v44, v89, v72
	v_cvt_f32_i32_e32 v46, v48
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v139, v41, v33
	v_mul_f32_e32 v33, v96, v73
	v_dual_fmac_f32 v142, v42, v45 :: v_dual_fmac_f32 v143, v44, v46
	v_dual_mul_f32 v42, v99, v73 :: v_dual_mul_f32 v41, v97, v73
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v137, v33, v35 :: v_dual_fmac_f32 v140, v42, v34
	v_mul_f32_e32 v34, v94, v73
	v_dual_fmac_f32 v138, v41, v36 :: v_dual_mul_f32 v33, v95, v73
	v_cvt_f32_i32_e32 v35, v38
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v34, v37
	v_mul_f32_e32 v34, v88, v73
	v_cvt_f32_i32_e32 v37, v39
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v136, v33, v35 :: v_dual_mul_f32 v33, v82, v80
	v_mul_f32_e32 v36, v89, v73
	v_cvt_f32_i32_e32 v38, v40
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v131, v33, v25
	s_wait_dscnt 0x2
	v_mul_f32_e32 v25, v82, v78
	v_dual_fmac_f32 v133, v34, v37 :: v_dual_fmac_f32 v134, v36, v38
	v_dual_mul_f32 v34, v82, v81 :: v_dual_mul_f32 v33, v82, v79
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v129, v25, v27 :: v_dual_fmac_f32 v132, v34, v26
	s_wait_dscnt 0x1
	v_mul_f32_e32 v26, v82, v76
	v_dual_fmac_f32 v130, v33, v28 :: v_dual_mul_f32 v25, v82, v77
	v_cvt_f32_i32_e32 v27, v30
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v26, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v82, v74
	v_cvt_f32_i32_e32 v29, v31
	v_fmac_f32_e32 v127, v25, v27
	v_dual_mul_f32 v25, v83, v80 :: v_dual_mul_f32 v28, v82, v75
	v_cvt_f32_i32_e32 v30, v32
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v122, v25, v17 :: v_dual_mul_f32 v17, v83, v78
	v_dual_fmac_f32 v125, v26, v29 :: v_dual_fmac_f32 v126, v28, v30
	v_dual_mul_f32 v26, v83, v81 :: v_dual_mul_f32 v25, v83, v79
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v120, v17, v19 :: v_dual_fmac_f32 v123, v26, v18
	v_dual_mul_f32 v18, v83, v76 :: v_dual_mul_f32 v17, v83, v77
	v_cvt_f32_i32_e32 v19, v22
	v_fmac_f32_e32 v121, v25, v20
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v118, v18, v21
	v_mul_f32_e32 v18, v83, v74
	v_cvt_f32_i32_e32 v21, v23
	v_fmac_f32_e32 v119, v17, v19
	v_dual_mul_f32 v17, v72, v80 :: v_dual_mul_f32 v20, v83, v75
	v_cvt_f32_i32_e32 v22, v24
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v114, v17, v9 :: v_dual_mul_f32 v9, v72, v78
	v_dual_fmac_f32 v116, v18, v21 :: v_dual_fmac_f32 v117, v20, v22
	v_dual_mul_f32 v18, v72, v81 :: v_dual_mul_f32 v17, v72, v79
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v112, v9, v11 :: v_dual_fmac_f32 v115, v18, v10
	v_dual_mul_f32 v10, v72, v76 :: v_dual_mul_f32 v9, v72, v77
	v_fmac_f32_e32 v113, v17, v12
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v11, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v110, v10, v13
	v_mul_f32_e32 v10, v72, v74
	v_cvt_f32_i32_e32 v13, v15
	v_dual_mul_f32 v15, v73, v80 :: v_dual_mul_f32 v12, v72, v75
	v_cvt_f32_i32_e32 v14, v16
	v_cvt_f32_i32_e32 v2, v2
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v106, v15, v1 :: v_dual_mul_f32 v1, v73, v78
	v_dual_mul_f32 v16, v73, v81 :: v_dual_fmac_f32 v109, v12, v14
	s_barrier_signal -1
	v_dual_fmac_f32 v108, v10, v13 :: v_dual_fmac_f32 v111, v9, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v107, v16, v2
	v_cvt_f32_i32_e32 v2, v3
	v_mul_f32_e32 v3, v73, v79
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_fmac_f32_e32 v104, v1, v2
	v_dual_mul_f32 v1, v73, v76 :: v_dual_mul_f32 v2, v73, v77
	v_dual_mul_f32 v9, v73, v74 :: v_dual_mul_f32 v10, v73, v75
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v151, v52, v54
	v_dual_fmac_f32 v105, v3, v4 :: v_dual_fmac_f32 v102, v1, v5
	v_fmac_f32_e32 v103, v2, v6
	v_fmac_f32_e32 v101, v9, v7
	v_fmac_f32_e32 v67, v10, v8
	s_xor_b32 s4, s28, -1
	s_mov_b32 s30, 1
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s29, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_118
.LBB1_121:                              ;   Parent Loop BB1_119 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s34, s30, 6
	s_mov_b32 s35, s5
	s_or_b32 s4, s30, s7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[8:9], s[34:35]
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s31
	v_add_co_u32 v3, s31, s34, v71
	s_mul_u64 s[36:37], s[36:37], 0x48
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_clause 0x1
	global_load_b64 v[7:8], v[3:4], off offset:40
	global_load_b64 v[80:81], v[1:2], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v1, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	s_clause 0x1
	global_load_b64 v[92:93], v[5:6], off offset:40
	global_load_b64 v[84:85], v[1:2], off offset:40
	v_add_nc_u32_e32 v82, s26, v169
	v_add_nc_u32_e32 v83, s27, v168
	ds_load_2addr_stride64_b64 v[72:75], v82 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v83 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v83 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	v_xor_b32_e32 v87, 0x88888888, v8
	v_xor_b32_e32 v86, 0x88888888, v7
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v91, 0x88888888, v81
	v_xor_b32_e32 v90, 0x88888888, v80
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v82 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v83 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v83 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v93, s0
	v_cndmask_b32_e64 v72, 0, v92, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_2addr_stride64_b64 v166, v[90:91], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v85, s1
	v_cndmask_b32_e64 v72, 0, v84, s1
	ds_store_2addr_stride64_b64 v167, v[86:87], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_123
; %bb.122:                              ;   in Loop: Header=BB1_121 Depth=2
	s_add_co_i32 s4, s4, 1
	s_xor_b32 s27, s30, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_add_co_i32 s4, s30, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[30:31], s[4:5], 0x88
	s_mov_b32 s37, s5
	s_lshl_b32 s36, s27, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	s_add_nc_u64 s[30:31], s[16:17], s[30:31]
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, s33, s34, v68
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	v_mad_co_i64_i32 v[80:81], null, 0x48, v170, s[34:35]
	v_add_co_u32 v82, vcc_lo, v65, s4
	v_add_co_ci_u32_e64 v73, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s30, v69
	v_add_co_u32 v76, s30, s30, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	s_lshl_b32 s4, s27, 2
	v_add_co_ci_u32_e64 v77, null, s31, 0, s30
	v_add_co_u32 v78, s30, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	v_add_co_u32 v80, vcc_lo, v80, v175
	s_clause 0x1
	global_load_b64 v[74:75], v[74:75], off offset:8
	global_load_b64 v[76:77], v[76:77], off offset:8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[92:93], v[72:73], off offset:8
	global_load_b64 v[84:85], v[78:79], off offset:8
	global_load_b32 v176, v[82:83], off
	global_load_b32 v177, v[80:81], off
	s_wait_loadcnt 0x5
	v_xor_b32_e32 v91, 0x88888888, v75
	s_wait_loadcnt 0x4
	v_xor_b32_e32 v87, 0x88888888, v77
	v_xor_b32_e32 v86, 0x88888888, v76
	v_xor_b32_e32 v90, 0x88888888, v74
.LBB1_123:                              ; %.preheader517.i
                                        ;   in Loop: Header=BB1_121 Depth=2
	v_add_nc_u32_e32 v88, 0, v169
	v_add_nc_u32_e32 v89, 0, v168
	s_xor_b32 s4, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s11, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v88 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v89 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v89 offset0:34 offset1:35
	s_cselect_b32 s27, s13, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v88
	v_add_nc_u32_e32 v80, 0x100, v89
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s27, v173
	v_add_nc_u32_e32 v72, s26, v174
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[98:99], v74 offset1:2
	ds_load_2addr_b32 v[96:97], v74 offset0:4 offset1:6
	ds_load_2addr_b32 v[94:95], v74 offset0:8 offset1:10
	ds_load_2addr_b32 v[88:89], v74 offset0:12 offset1:14
	ds_load_2addr_b32 v[82:83], v72 offset1:32
	ds_load_2addr_b32 v[72:73], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[80:81], v74 offset0:32 offset1:34
	ds_load_2addr_b32 v[78:79], v74 offset0:36 offset1:38
	ds_load_2addr_b32 v[76:77], v74 offset0:40 offset1:42
	ds_load_2addr_b32 v[74:75], v74 offset0:44 offset1:46
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_120
; %bb.124:                              ; %.preheader518.i
                                        ;   in Loop: Header=BB1_121 Depth=2
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v172, v176
	s_and_b32 s4, s29, exec_lo
	s_cselect_b32 s4, s11, 0x3400
	v_cndmask_b32_e64 v93, 0, v93, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v180, s4, v171
	v_cvt_f32_f16_e64 v178, v178.l
	s_cselect_b32 s4, s13, 0x3c00
	v_cndmask_b32_e64 v92, 0, v92, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v179, 0, v177, s2
	v_cndmask_b32_e64 v85, 0, v85, s1
	v_cndmask_b32_e64 v84, 0, v84, s1
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v181, s4, v171
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v166, v[92:93], v[90:91] offset1:8
	ds_store_2addr_stride64_b64 v167, v[84:85], v[86:87] offset1:8
	ds_store_b32 v180, v179
	ds_store_b32 v181, v178
	s_branch .LBB1_120
.LBB1_125:                              ; %._crit_edge585.i
	s_add_co_i32 s0, s22, 0x80
	s_ashr_i32 s13, s12, 31
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_le_i32 s0, s12
	v_mul_u32_u24_e32 v1, 0x500, v100
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s24, 0x80
	v_lshlrev_b32_e32 v3, 2, v124
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
	s_cbranch_vccnz .LBB1_216
; %bb.126:                              ; %.preheader514.i
	v_mad_u32_u24 v0, 0x280, v141, v3
	v_or_b32_e32 v6, s24, v2
	v_mul_u32_u24_e32 v1, 0x50, v124
	v_lshlrev_b32_e32 v4, 2, v2
	ds_store_2addr_b32 v0, v158, v165 offset1:20
	ds_store_2addr_b32 v0, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v0, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v0, v159, v160 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s22, v124
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
	s_cbranch_execz .LBB1_128
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
.LBB1_128:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, 64, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v7
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_130
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
.LBB1_130:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v5
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_132
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
.LBB1_132:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_134
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
.LBB1_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v5
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_136
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
.LBB1_136:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_138
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
.LBB1_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v5, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v5
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_140
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
.LBB1_140:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v5, 0x280, v141
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_142
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
.LBB1_142:                              ; %.preheader512.1.i
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
	ds_store_2addr_b32 v5, v156, v157 offset1:20
	ds_store_2addr_b32 v5, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v153, v152 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v150, v151 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_144
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
.LBB1_144:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v9, 0x50, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v9
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_261
; %bb.145:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_262
.LBB1_146:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_263
.LBB1_147:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_264
.LBB1_148:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_265
.LBB1_149:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_266
.LBB1_150:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_152
.LBB1_151:
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
.LBB1_152:                              ; %.preheader512.2.i
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
	ds_store_2addr_b32 v5, v148, v149 offset1:20
	ds_store_2addr_b32 v5, v146, v147 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v144, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v142, v143 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_154
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
.LBB1_154:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v11, 0x60, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v11
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_267
; %bb.155:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_268
.LBB1_156:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_269
.LBB1_157:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_270
.LBB1_158:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_271
.LBB1_159:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_272
.LBB1_160:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_162
.LBB1_161:
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
.LBB1_162:                              ; %.preheader512.3.i
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
	ds_store_2addr_b32 v5, v139, v140 offset1:20
	ds_store_2addr_b32 v5, v137, v138 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v135, v136 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v133, v134 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB1_164
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
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v13, 0x70, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v13
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_273
; %bb.165:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB1_274
.LBB1_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_275
.LBB1_167:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_276
.LBB1_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_277
.LBB1_169:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_278
.LBB1_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_172
.LBB1_171:
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
.LBB1_172:                              ; %.preheader513.1.i
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
	ds_store_2addr_b32 v5, v131, v132 offset1:20
	ds_store_2addr_b32 v5, v129, v130 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v128, v127 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v125, v126 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_174
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
.LBB1_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_176
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
.LBB1_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v14, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v14
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_178
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
.LBB1_178:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_180
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
.LBB1_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v14, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v14
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_182
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
.LBB1_182:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_184
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
.LBB1_184:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v14, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v14
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB1_186
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
.LBB1_186:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_188
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
.LBB1_188:                              ; %.preheader512.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v122, v123 offset1:20
	ds_store_2addr_b32 v5, v120, v121 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v118, v119 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v116, v117 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_279
; %bb.189:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_280
.LBB1_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_281
.LBB1_191:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_282
.LBB1_192:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_283
.LBB1_193:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB1_284
.LBB1_194:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_285
.LBB1_195:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_197
.LBB1_196:
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
.LBB1_197:                              ; %.preheader512.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v114, v115 offset1:20
	ds_store_2addr_b32 v5, v112, v113 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v110, v111 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v108, v109 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_286
; %bb.198:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_287
.LBB1_199:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_288
.LBB1_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_289
.LBB1_201:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_290
.LBB1_202:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_291
.LBB1_203:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_292
.LBB1_204:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_206
.LBB1_205:
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
.LBB1_206:                              ; %.preheader512.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v5, v106, v107 offset1:20
	ds_store_2addr_b32 v5, v104, v105 offset0:40 offset1:60
	ds_store_2addr_b32 v5, v102, v103 offset0:80 offset1:100
	ds_store_2addr_b32 v5, v101, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_293
; %bb.207:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_294
.LBB1_208:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_295
.LBB1_209:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_296
.LBB1_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_297
.LBB1_211:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_298
.LBB1_212:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_299
.LBB1_213:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_215
.LBB1_214:
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
.LBB1_215:                              ; %.loopexit.loopexit618.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB1_216:                              ; %Flow1681
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_218
; %bb.217:                              ; %.preheader511.i
	v_mad_u32_u24 v9, 0x280, v141, v3
	v_mul_u32_u24_e32 v0, 0x50, v124
	v_lshlrev_b32_e32 v1, 2, v2
	v_mul_lo_u32 v2, s12, v2
	s_ashr_i32 s25, s24, 31
	ds_store_2addr_b32 v9, v158, v165 offset1:20
	ds_store_2addr_b32 v9, v163, v164 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v161, v162 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v159, v160 offset0:120 offset1:140
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
	v_add_lshl_u32 v11, v2, v124, 2
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
	ds_store_2addr_b32 v9, v156, v157 offset1:20
	ds_store_2addr_b32 v9, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v153, v152 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v150, v151 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v148, v149 offset1:20
	ds_store_2addr_b32 v9, v146, v147 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v144, v145 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v142, v143 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v139, v140 offset1:20
	ds_store_2addr_b32 v9, v137, v138 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v135, v136 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v133, v134 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v131, v132 offset1:20
	ds_store_2addr_b32 v9, v129, v130 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v128, v127 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v125, v126 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v122, v123 offset1:20
	ds_store_2addr_b32 v9, v120, v121 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v118, v119 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v116, v117 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v114, v115 offset1:20
	ds_store_2addr_b32 v9, v112, v113 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v110, v111 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v108, v109 offset0:120 offset1:140
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
	ds_store_2addr_b32 v9, v106, v107 offset1:20
	ds_store_2addr_b32 v9, v104, v105 offset0:40 offset1:60
	ds_store_2addr_b32 v9, v102, v103 offset0:80 offset1:100
	ds_store_2addr_b32 v9, v101, v67 offset0:120 offset1:140
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
.LBB1_218:                              ; %Flow1682
	s_mov_b32 s0, -1
.LBB1_219:                              ; %Flow1690
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_221
; %bb.220:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB1_221:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB1_222:
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
	s_cbranch_execz .LBB1_38
.LBB1_223:
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
	s_cbranch_execz .LBB1_39
.LBB1_224:
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
	s_cbranch_execz .LBB1_40
.LBB1_225:
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
	s_cbranch_execz .LBB1_41
.LBB1_226:
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
	s_cbranch_execz .LBB1_42
.LBB1_227:
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
	s_cbranch_execnz .LBB1_43
	s_branch .LBB1_44
.LBB1_228:
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
	s_cbranch_execz .LBB1_48
.LBB1_229:
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
	s_cbranch_execz .LBB1_49
.LBB1_230:
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
	s_cbranch_execz .LBB1_50
.LBB1_231:
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
	s_cbranch_execz .LBB1_51
.LBB1_232:
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
	s_cbranch_execz .LBB1_52
.LBB1_233:
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
	s_cbranch_execnz .LBB1_53
	s_branch .LBB1_54
.LBB1_234:
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
	s_cbranch_execz .LBB1_58
.LBB1_235:
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
	s_cbranch_execz .LBB1_59
.LBB1_236:
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
	s_cbranch_execz .LBB1_60
.LBB1_237:
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
	s_cbranch_execz .LBB1_61
.LBB1_238:
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
	s_cbranch_execz .LBB1_62
.LBB1_239:
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
	s_cbranch_execnz .LBB1_63
	s_branch .LBB1_64
.LBB1_240:
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
	s_cbranch_execz .LBB1_82
.LBB1_241:
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
	s_cbranch_execz .LBB1_83
.LBB1_242:
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
	s_cbranch_execz .LBB1_84
.LBB1_243:
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
	s_cbranch_execz .LBB1_85
.LBB1_244:
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
	s_cbranch_execz .LBB1_86
.LBB1_245:
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
	s_cbranch_execz .LBB1_87
.LBB1_246:
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
	s_cbranch_execnz .LBB1_88
	s_branch .LBB1_89
.LBB1_247:
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
	s_cbranch_execz .LBB1_91
.LBB1_248:
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
	s_cbranch_execz .LBB1_92
.LBB1_249:
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
	s_cbranch_execz .LBB1_93
.LBB1_250:
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
	s_cbranch_execz .LBB1_94
.LBB1_251:
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
	s_cbranch_execz .LBB1_95
.LBB1_252:
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
	s_cbranch_execz .LBB1_96
.LBB1_253:
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
	s_cbranch_execnz .LBB1_97
	s_branch .LBB1_98
.LBB1_254:
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
	s_cbranch_execz .LBB1_100
.LBB1_255:
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
	s_cbranch_execz .LBB1_101
.LBB1_256:
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
	s_cbranch_execz .LBB1_102
.LBB1_257:
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
	s_cbranch_execz .LBB1_103
.LBB1_258:
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
	s_cbranch_execz .LBB1_104
.LBB1_259:
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
	s_cbranch_execz .LBB1_105
.LBB1_260:
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
	s_cbranch_execnz .LBB1_106
	s_branch .LBB1_107
.LBB1_261:
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
	s_cbranch_execz .LBB1_146
.LBB1_262:
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
	s_cbranch_execz .LBB1_147
.LBB1_263:
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
	s_cbranch_execz .LBB1_148
.LBB1_264:
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
	s_cbranch_execz .LBB1_149
.LBB1_265:
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
	s_cbranch_execz .LBB1_150
.LBB1_266:
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
	s_cbranch_execnz .LBB1_151
	s_branch .LBB1_152
.LBB1_267:
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
	s_cbranch_execz .LBB1_156
.LBB1_268:
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
	s_cbranch_execz .LBB1_157
.LBB1_269:
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
	s_cbranch_execz .LBB1_158
.LBB1_270:
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
	s_cbranch_execz .LBB1_159
.LBB1_271:
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
	s_cbranch_execz .LBB1_160
.LBB1_272:
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
	s_cbranch_execnz .LBB1_161
	s_branch .LBB1_162
.LBB1_273:
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
	s_cbranch_execz .LBB1_166
.LBB1_274:
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
	s_cbranch_execz .LBB1_167
.LBB1_275:
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
	s_cbranch_execz .LBB1_168
.LBB1_276:
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
	s_cbranch_execz .LBB1_169
.LBB1_277:
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
	s_cbranch_execz .LBB1_170
.LBB1_278:
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
	s_cbranch_execnz .LBB1_171
	s_branch .LBB1_172
.LBB1_279:
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
	s_cbranch_execz .LBB1_190
.LBB1_280:
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
	s_cbranch_execz .LBB1_191
.LBB1_281:
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
	s_cbranch_execz .LBB1_192
.LBB1_282:
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
	s_cbranch_execz .LBB1_193
.LBB1_283:
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
	s_cbranch_execz .LBB1_194
.LBB1_284:
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
	s_cbranch_execz .LBB1_195
.LBB1_285:
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
	s_cbranch_execnz .LBB1_196
	s_branch .LBB1_197
.LBB1_286:
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
	s_cbranch_execz .LBB1_199
.LBB1_287:
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
	s_cbranch_execz .LBB1_200
.LBB1_288:
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
	s_cbranch_execz .LBB1_201
.LBB1_289:
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
	s_cbranch_execz .LBB1_202
.LBB1_290:
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
	s_cbranch_execz .LBB1_203
.LBB1_291:
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
	s_cbranch_execz .LBB1_204
.LBB1_292:
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
	s_cbranch_execnz .LBB1_205
	s_branch .LBB1_206
.LBB1_293:
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
	s_cbranch_execz .LBB1_208
.LBB1_294:
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
	s_cbranch_execz .LBB1_209
.LBB1_295:
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
	s_cbranch_execz .LBB1_210
.LBB1_296:
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
	s_cbranch_execz .LBB1_211
.LBB1_297:
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
	s_cbranch_execz .LBB1_212
.LBB1_298:
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
	s_cbranch_execz .LBB1_213
.LBB1_299:
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
	s_cbranch_execnz .LBB1_214
	s_branch .LBB1_215
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
		.amdhsa_next_free_vgpr 182
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_vgpr, 182
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
; codeLenInByte = 33080
; TotalNumSgprs: 43
; NumVgprs: 182
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 182
; Occupancy: 8
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
	s_cbranch_vccnz .LBB2_108
; %bb.1:                                ; %.preheader530.i
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
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s18, v2
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
.LBB2_4:                                ; %.preheader525.loopexit.i
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
.LBB2_5:                                ; %Flow779
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v100, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s18, v11
	v_dual_mov_b32 v101, 0 :: v_dual_add_nc_u32 v4, s16, v11
	v_dual_mov_b32 v106, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v102, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s3, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s3, v5
	v_dual_mov_b32 v154, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s2, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s2, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v165, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v126, 0
	v_or_b32_e32 v15, 8, v165
	v_and_or_b32 v16, v165, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v160, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v132, 0 :: v_dual_and_b32 v139, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v121, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v164, v0, 4, 1
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v136, 0
	v_add_nc_u32_e32 v166, 0, v15
	v_add_nc_u32_e32 v167, 0, v11
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v156, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v163, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v6, 0x88888888, v6
	v_xor_b32_e32 v5, 0x88888888, v5
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v8, 0, v8, s1
	v_cndmask_b32_e64 v7, 0, v7, s1
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v10, 0x88888888, v10
	v_xor_b32_e32 v9, 0x88888888, v9
	ds_store_2addr_stride64_b64 v166, v[3:4], v[5:6] offset1:8
	ds_store_2addr_stride64_b64 v167, v[7:8], v[9:10] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_14
; %bb.6:                                ; %.preheader524.lr.ph.i
	v_dual_mov_b32 v156, 0 :: v_dual_add_nc_u32 v3, s16, v2
	v_dual_mov_b32 v177, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s3, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v159, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_dual_mov_b32 v163, 0 :: v_dual_add_nc_u32 v8, s18, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[65:66], null, s2, v7, s[4:5]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v9, 6, v164
	v_dual_mov_b32 v161, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v157, 0 :: v_dual_lshlrev_b32 v10, 2, v1
	v_dual_mov_b32 v162, 0 :: v_dual_lshlrev_b32 v11, 8, v5
	v_lshl_or_b32 v169, v6, 11, v4
	v_lshl_or_b32 v170, v5, 10, v4
	v_dual_mov_b32 v129, 0 :: v_dual_lshlrev_b32 v4, 9, v6
	v_dual_mov_b32 v160, 0 :: v_dual_lshlrev_b32 v5, 3, v139
	v_mad_co_u64_u32 v[66:67], null, s2, v7, v[66:67]
	v_min_i32_e32 v168, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v171, 0, v2, v10
	v_cmp_gt_i32_e64 s3, s12, v3
	v_dual_mov_b32 v127, 0 :: v_dual_lshlrev_b32 v172, 4, v1
	v_add3_u32 v173, 0, v11, v9
	v_add3_u32 v174, 0, v4, v5
	v_dual_mov_b32 v158, 0 :: v_dual_lshlrev_b32 v175, 2, v1
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v100, 0
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s19, 0x3800
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	s_branch .LBB2_8
.LBB2_7:                                ;   in Loop: Header=BB2_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_8:                                ; %.preheader524.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_10 Depth 2
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
	s_branch .LBB2_10
.LBB2_9:                                ;   in Loop: Header=BB2_10 Depth=2
	s_wait_loadcnt_dscnt 0x205
	v_dual_mul_f32 v84, v98, v82 :: v_dual_mul_f32 v85, v99, v82
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v156, v84, v57 :: v_dual_fmac_f32 v163, v85, v58
	v_dual_mul_f32 v57, v96, v82 :: v_dual_mul_f32 v84, v97, v82
	v_mul_f32_e32 v58, v94, v82
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_dual_fmac_f32 v161, v57, v59 :: v_dual_fmac_f32 v162, v84, v60
	v_mul_f32_e32 v57, v95, v82
	v_cvt_f32_i32_e32 v59, v62
	v_dual_fmac_f32 v159, v58, v61 :: v_dual_mul_f32 v58, v88, v82
	v_mul_f32_e32 v60, v89, v82
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_fmac_f32_e32 v160, v57, v59
	v_mul_f32_e32 v57, v98, v83
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v157, v58, v61 :: v_dual_fmac_f32 v158, v60, v62
	v_mul_f32_e32 v58, v99, v83
	v_dual_fmac_f32 v154, v57, v49 :: v_dual_mul_f32 v49, v96, v83
	v_mul_f32_e32 v57, v97, v83
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v58, v50
	v_mul_f32_e32 v50, v94, v83
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v41, v41
	v_fmac_f32_e32 v153, v57, v52
	v_fmac_f32_e32 v152, v49, v51
	v_mul_f32_e32 v49, v95, v83
	v_cvt_f32_i32_e32 v51, v54
	v_dual_fmac_f32 v151, v50, v53 :: v_dual_mul_f32 v50, v88, v83
	v_mul_f32_e32 v52, v89, v83
	v_cvt_f32_i32_e32 v53, v55
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v150, v49, v51 :: v_dual_mul_f32 v49, v98, v72
	v_cvt_f32_i32_e32 v54, v56
	v_cvt_f32_i32_e32 v42, v42
	v_dual_fmac_f32 v149, v50, v53 :: v_dual_mul_f32 v50, v99, v72
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v146, v49, v41 :: v_dual_mul_f32 v41, v96, v72
	v_cvt_f32_i32_e32 v43, v43
	v_dual_fmac_f32 v148, v52, v54 :: v_dual_mul_f32 v49, v97, v72
	v_cvt_f32_i32_e32 v44, v44
	v_fmac_f32_e32 v147, v50, v42
	v_mul_f32_e32 v42, v94, v72
	v_cvt_f32_i32_e32 v45, v45
	v_dual_fmac_f32 v144, v41, v43 :: v_dual_mul_f32 v41, v95, v72
	v_cvt_f32_i32_e32 v43, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v145, v49, v44 :: v_dual_fmac_f32 v142, v42, v45
	v_mul_f32_e32 v42, v88, v72
	v_cvt_f32_i32_e32 v45, v47
	v_fmac_f32_e32 v143, v41, v43
	v_mul_f32_e32 v41, v98, v73
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v44, v89, v72
	v_cvt_f32_i32_e32 v46, v48
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v137, v41, v33
	v_mul_f32_e32 v33, v96, v73
	v_dual_fmac_f32 v140, v42, v45 :: v_dual_fmac_f32 v141, v44, v46
	v_dual_mul_f32 v42, v99, v73 :: v_dual_mul_f32 v41, v97, v73
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v135, v33, v35 :: v_dual_fmac_f32 v138, v42, v34
	v_mul_f32_e32 v34, v94, v73
	v_dual_fmac_f32 v136, v41, v36 :: v_dual_mul_f32 v33, v95, v73
	v_cvt_f32_i32_e32 v35, v38
	v_cvt_f32_i32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v133, v34, v37
	v_mul_f32_e32 v34, v88, v73
	v_cvt_f32_i32_e32 v37, v39
	s_wait_dscnt 0x3
	v_dual_fmac_f32 v134, v33, v35 :: v_dual_mul_f32 v33, v82, v80
	v_mul_f32_e32 v36, v89, v73
	v_cvt_f32_i32_e32 v38, v40
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_fmac_f32_e32 v129, v33, v25
	s_wait_dscnt 0x2
	v_mul_f32_e32 v25, v82, v78
	v_dual_fmac_f32 v131, v34, v37 :: v_dual_fmac_f32 v132, v36, v38
	v_dual_mul_f32 v34, v82, v81 :: v_dual_mul_f32 v33, v82, v79
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v127, v25, v27 :: v_dual_fmac_f32 v130, v34, v26
	s_wait_dscnt 0x1
	v_mul_f32_e32 v26, v82, v76
	v_dual_fmac_f32 v128, v33, v28 :: v_dual_mul_f32 v25, v82, v77
	v_cvt_f32_i32_e32 v27, v30
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v126, v26, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v82, v74
	v_cvt_f32_i32_e32 v29, v31
	v_fmac_f32_e32 v125, v25, v27
	v_dual_mul_f32 v25, v83, v80 :: v_dual_mul_f32 v28, v82, v75
	v_cvt_f32_i32_e32 v30, v32
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v121, v25, v17
	v_dual_fmac_f32 v123, v26, v29 :: v_dual_fmac_f32 v124, v28, v30
	v_dual_mul_f32 v26, v83, v81 :: v_dual_mul_f32 v17, v83, v78
	v_mul_f32_e32 v25, v83, v79
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v122, v26, v18
	v_mul_f32_e32 v18, v83, v76
	v_fmac_f32_e32 v120, v17, v19
	v_fmac_f32_e32 v119, v25, v20
	v_mul_f32_e32 v17, v83, v77
	v_cvt_f32_i32_e32 v19, v22
	v_dual_fmac_f32 v117, v18, v21 :: v_dual_mul_f32 v18, v83, v74
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v20, v83, v75
	v_cvt_f32_i32_e32 v22, v24
	v_dual_fmac_f32 v118, v17, v19 :: v_dual_mul_f32 v17, v72, v80
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v115, v18, v21
	v_mul_f32_e32 v18, v72, v81
	v_cvt_f32_i32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v116, v20, v22 :: v_dual_fmac_f32 v113, v17, v9
	v_mul_f32_e32 v9, v72, v78
	v_dual_mul_f32 v17, v72, v79 :: v_dual_fmac_f32 v114, v18, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_mul_f32_e32 v10, v72, v76
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	v_fmac_f32_e32 v111, v9, v11
	v_dual_fmac_f32 v112, v17, v12 :: v_dual_mul_f32 v9, v72, v77
	v_cvt_f32_i32_e32 v11, v14
	v_dual_fmac_f32 v109, v10, v13 :: v_dual_mul_f32 v10, v72, v74
	v_mul_f32_e32 v12, v72, v75
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v73, v80 :: v_dual_mul_f32 v16, v73, v81
	v_cvt_f32_i32_e32 v2, v2
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v108, v12, v14 :: v_dual_fmac_f32 v105, v15, v1
	v_mul_f32_e32 v1, v73, v78
	v_fmac_f32_e32 v106, v16, v2
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v110, v9, v11 :: v_dual_fmac_f32 v107, v10, v13
	v_mul_f32_e32 v3, v73, v79
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v1, v2
	v_dual_mul_f32 v1, v73, v76 :: v_dual_mul_f32 v2, v73, v77
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_dual_mul_f32 v9, v73, v74 :: v_dual_mul_f32 v10, v73, v75
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v104, v3, v4 :: v_dual_fmac_f32 v101, v1, v5
	v_fmac_f32_e32 v102, v2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v100, v9, v7 :: v_dual_fmac_f32 v67, v10, v8
	s_xor_b32 s8, s28, -1
	s_mov_b32 s30, 1
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s29, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_7
.LBB2_10:                               ;   Parent Loop BB2_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s34, s30, 6
	s_mov_b32 s35, s9
	s_or_b32 s8, s30, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_mul_u64 s[36:37], s[8:9], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s31
	v_add_co_u32 v3, s31, s34, v71
	s_mul_u64 s[36:37], s[36:37], 0x48
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[6:7], s[36:37]
	s_clause 0x1
	global_load_b64 v[7:8], v[3:4], off offset:40
	global_load_b64 v[80:81], v[1:2], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v1, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	s_clause 0x1
	global_load_b64 v[92:93], v[5:6], off offset:40
	global_load_b64 v[84:85], v[1:2], off offset:40
	v_add_nc_u32_e32 v82, s26, v170
	v_add_nc_u32_e32 v83, s27, v169
	ds_load_2addr_stride64_b64 v[72:75], v82 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v83 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v83 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	v_xor_b32_e32 v87, 0x88888888, v8
	v_xor_b32_e32 v86, 0x88888888, v7
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v91, 0x88888888, v81
	v_xor_b32_e32 v90, 0x88888888, v80
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v82 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v83 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v83 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v93, s0
	v_cndmask_b32_e64 v72, 0, v92, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_2addr_stride64_b64 v166, v[90:91], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v85, s1
	v_cndmask_b32_e64 v72, 0, v84, s1
	ds_store_2addr_stride64_b64 v167, v[86:87], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
	s_add_co_i32 s8, s8, 1
	s_xor_b32 s27, s30, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s30, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[30:31], s[8:9], 0x88
	s_mov_b32 s37, s9
	s_lshl_b32 s36, s27, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_add_nc_u64 s[30:31], s[4:5], s[30:31]
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, s33, s34, v68
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	v_mad_co_i64_i32 v[80:81], null, 0x48, v168, s[34:35]
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_ci_u32_e64 v73, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s30, v69
	v_add_co_u32 v76, s30, s30, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	s_lshl_b32 s8, s27, 2
	v_add_co_ci_u32_e64 v77, null, s31, 0, s30
	v_add_co_u32 v78, s30, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	v_add_co_u32 v80, vcc_lo, v80, v175
	s_clause 0x1
	global_load_b64 v[74:75], v[74:75], off offset:8
	global_load_b64 v[76:77], v[76:77], off offset:8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[92:93], v[72:73], off offset:8
	global_load_b64 v[84:85], v[78:79], off offset:8
	global_load_b32 v176, v[82:83], off
	global_load_b32 v177, v[80:81], off
	s_wait_loadcnt 0x5
	v_xor_b32_e32 v91, 0x88888888, v75
	s_wait_loadcnt 0x4
	v_xor_b32_e32 v87, 0x88888888, v77
	v_xor_b32_e32 v86, 0x88888888, v76
	v_xor_b32_e32 v90, 0x88888888, v74
.LBB2_12:                               ; %.preheader522.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	v_add_nc_u32_e32 v88, 0, v170
	v_add_nc_u32_e32 v89, 0, v169
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v88 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v89 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v89 offset0:34 offset1:35
	s_cselect_b32 s27, s19, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v88
	v_add_nc_u32_e32 v80, 0x100, v89
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s27, v173
	v_add_nc_u32_e32 v72, s26, v174
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[98:99], v74 offset1:2
	ds_load_2addr_b32 v[96:97], v74 offset0:4 offset1:6
	ds_load_2addr_b32 v[94:95], v74 offset0:8 offset1:10
	ds_load_2addr_b32 v[88:89], v74 offset0:12 offset1:14
	ds_load_2addr_b32 v[82:83], v72 offset1:32
	ds_load_2addr_b32 v[72:73], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[80:81], v74 offset0:32 offset1:34
	ds_load_2addr_b32 v[78:79], v74 offset0:36 offset1:38
	ds_load_2addr_b32 v[76:77], v74 offset0:40 offset1:42
	ds_load_2addr_b32 v[74:75], v74 offset0:44 offset1:46
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_9
; %bb.13:                               ; %.preheader523.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v172, v176
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v93, 0, v93, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v180, s8, v171
	v_cvt_f32_f16_e64 v178, v178.l
	s_cselect_b32 s8, s19, 0x3c00
	v_cndmask_b32_e64 v92, 0, v92, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v179, 0, v177, s2
	v_cndmask_b32_e64 v85, 0, v85, s1
	v_cndmask_b32_e64 v84, 0, v84, s1
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v181, s8, v171
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v166, v[92:93], v[90:91] offset1:8
	ds_store_2addr_stride64_b64 v167, v[84:85], v[86:87] offset1:8
	ds_store_b32 v180, v179
	ds_store_b32 v181, v178
	s_branch .LBB2_9
.LBB2_14:                               ; %._crit_edge590.i
	v_mul_u32_u24_e32 v1, 0x500, v165
	v_lshlrev_b32_e32 v2, 2, v139
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
	v_mul_u32_u24_e32 v3, 0x50, v139
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x280, v164, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_105
; %bb.15:                               ; %.preheader519.i
	ds_store_2addr_b32 v2, v156, v163 offset1:20
	ds_store_2addr_b32 v2, v161, v162 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v159, v160 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v157, v158 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v139
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
	s_cbranch_execz .LBB2_17
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
.LBB2_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v9, 64, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v9
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_19
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
.LBB2_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v10
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_21
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
.LBB2_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_23
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
.LBB2_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v10
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_25
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
.LBB2_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_27
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
.LBB2_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v10, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v10
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_29
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
.LBB2_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v164
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_31
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
.LBB2_31:                               ; %.preheader517.1.i
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
	ds_store_2addr_b32 v7, v154, v155 offset1:20
	ds_store_2addr_b32 v7, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v151, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v149, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_33
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
.LBB2_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v11, 0x50, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v11
	s_and_b32 s3, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_109
; %bb.34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_110
.LBB2_35:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_111
.LBB2_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_112
.LBB2_37:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_113
.LBB2_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB2_114
.LBB2_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_41
.LBB2_40:
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
.LBB2_41:                               ; %.preheader517.2.i
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
	ds_store_2addr_b32 v7, v146, v147 offset1:20
	ds_store_2addr_b32 v7, v144, v145 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v143 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v140, v141 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_43
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
.LBB2_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v13, 0x60, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v13
	s_and_b32 s5, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_115
; %bb.44:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_116
.LBB2_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_117
.LBB2_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_118
.LBB2_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_119
.LBB2_48:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB2_120
.LBB2_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB2_51
.LBB2_50:
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
.LBB2_51:                               ; %.preheader517.3.i
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
	ds_store_2addr_b32 v7, v137, v138 offset1:20
	ds_store_2addr_b32 v7, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v131, v132 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB2_53
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
.LBB2_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v15, 0x70, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v15
	s_and_b32 s7, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_121
; %bb.54:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s7
	s_cbranch_execnz .LBB2_122
.LBB2_55:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_123
.LBB2_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_124
.LBB2_57:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_125
.LBB2_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB2_126
.LBB2_59:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB2_61
.LBB2_60:
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
.LBB2_61:                               ; %.preheader518.1.i
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
	ds_store_2addr_b32 v7, v129, v130 offset1:20
	ds_store_2addr_b32 v7, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v126, v125 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v123, v124 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_63
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
.LBB2_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_65
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
.LBB2_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v16, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v16
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_67
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
.LBB2_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_69
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
.LBB2_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v16, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v16
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_71
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
.LBB2_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_73
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
.LBB2_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v16, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v16
	s_and_b32 s14, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s14
	s_cbranch_execz .LBB2_75
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
.LBB2_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_77
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
.LBB2_77:                               ; %.preheader517.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v120, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v115, v116 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_127
; %bb.78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_128
.LBB2_79:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_129
.LBB2_80:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_130
.LBB2_81:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_131
.LBB2_82:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execnz .LBB2_132
.LBB2_83:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_133
.LBB2_84:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_86
.LBB2_85:
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
.LBB2_86:                               ; %.preheader517.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v113, v114 offset1:20
	ds_store_2addr_b32 v7, v111, v112 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v109, v110 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v107, v108 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_134
; %bb.87:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_135
.LBB2_88:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_136
.LBB2_89:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_137
.LBB2_90:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_138
.LBB2_91:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_139
.LBB2_92:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_140
.LBB2_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_95
.LBB2_94:
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
.LBB2_95:                               ; %.preheader517.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v105, v106 offset1:20
	ds_store_2addr_b32 v7, v103, v104 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v101, v102 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v100, v67 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_141
; %bb.96:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_142
.LBB2_97:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_143
.LBB2_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_144
.LBB2_99:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_145
.LBB2_100:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_146
.LBB2_101:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB2_147
.LBB2_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_104
.LBB2_103:
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
.LBB2_104:                              ; %.loopexit.loopexit623.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0
	s_barrier_wait -1
.LBB2_105:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_107
; %bb.106:                              ; %.preheader516.i
	ds_store_2addr_b32 v2, v156, v163 offset1:20
	ds_store_2addr_b32 v2, v161, v162 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v159, v160 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v157, v158 offset0:120 offset1:140
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
	v_add_lshl_u32 v5, v0, v139, 2
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
	ds_store_2addr_b32 v2, v154, v155 offset1:20
	ds_store_2addr_b32 v2, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v151, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v149, v148 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v146, v147 offset1:20
	ds_store_2addr_b32 v2, v144, v145 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v143 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v140, v141 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v137, v138 offset1:20
	ds_store_2addr_b32 v2, v135, v136 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v133, v134 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v131, v132 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v129, v130 offset1:20
	ds_store_2addr_b32 v2, v127, v128 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v126, v125 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v123, v124 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v120, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v115, v116 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v113, v114 offset1:20
	ds_store_2addr_b32 v2, v111, v112 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v109, v110 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v107, v108 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v105, v106 offset1:20
	ds_store_2addr_b32 v2, v103, v104 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v101, v102 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v100, v67 offset0:120 offset1:140
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
.LBB2_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB2_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB2_109:
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
	s_cbranch_execz .LBB2_35
.LBB2_110:
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
	s_cbranch_execz .LBB2_36
.LBB2_111:
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
	s_cbranch_execz .LBB2_37
.LBB2_112:
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
	s_cbranch_execz .LBB2_38
.LBB2_113:
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
	s_cbranch_execz .LBB2_39
.LBB2_114:
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
	s_cbranch_execnz .LBB2_40
	s_branch .LBB2_41
.LBB2_115:
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
	s_cbranch_execz .LBB2_45
.LBB2_116:
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
	s_cbranch_execz .LBB2_46
.LBB2_117:
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
	s_cbranch_execz .LBB2_47
.LBB2_118:
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
	s_cbranch_execz .LBB2_48
.LBB2_119:
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
	s_cbranch_execz .LBB2_49
.LBB2_120:
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
	s_cbranch_execnz .LBB2_50
	s_branch .LBB2_51
.LBB2_121:
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
	s_cbranch_execz .LBB2_55
.LBB2_122:
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
	s_cbranch_execz .LBB2_56
.LBB2_123:
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
	s_cbranch_execz .LBB2_57
.LBB2_124:
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
	s_cbranch_execz .LBB2_58
.LBB2_125:
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
	s_cbranch_execz .LBB2_59
.LBB2_126:
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
	s_cbranch_execnz .LBB2_60
	s_branch .LBB2_61
.LBB2_127:
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
	s_cbranch_execz .LBB2_79
.LBB2_128:
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
	s_cbranch_execz .LBB2_80
.LBB2_129:
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
	s_cbranch_execz .LBB2_81
.LBB2_130:
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
	s_cbranch_execz .LBB2_82
.LBB2_131:
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
	s_cbranch_execz .LBB2_83
.LBB2_132:
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
	s_cbranch_execz .LBB2_84
.LBB2_133:
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
	s_cbranch_execnz .LBB2_85
	s_branch .LBB2_86
.LBB2_134:
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
	s_cbranch_execz .LBB2_88
.LBB2_135:
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
	s_cbranch_execz .LBB2_89
.LBB2_136:
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
	s_cbranch_execz .LBB2_90
.LBB2_137:
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
	s_cbranch_execz .LBB2_91
.LBB2_138:
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
	s_cbranch_execz .LBB2_92
.LBB2_139:
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
	s_cbranch_execz .LBB2_93
.LBB2_140:
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
	s_cbranch_execnz .LBB2_94
	s_branch .LBB2_95
.LBB2_141:
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
	s_cbranch_execz .LBB2_97
.LBB2_142:
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
	s_cbranch_execz .LBB2_98
.LBB2_143:
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
	s_cbranch_execz .LBB2_99
.LBB2_144:
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
	s_cbranch_execz .LBB2_100
.LBB2_145:
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
	s_cbranch_execz .LBB2_101
.LBB2_146:
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
	s_cbranch_execz .LBB2_102
.LBB2_147:
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
	s_cbranch_execnz .LBB2_103
	s_branch .LBB2_104
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
		.amdhsa_next_free_vgpr 182
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_vgpr, 182
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
; codeLenInByte = 17680
; TotalNumSgprs: 40
; NumVgprs: 182
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 182
; Occupancy: 8
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
	s_cbranch_vccnz .LBB3_108
; %bb.1:                                ; %.preheader525.i
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
.LBB3_4:                                ; %.preheader520.loopexit.i
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
.LBB3_5:                                ; %Flow779
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v11, 2, v0
	v_dual_mov_b32 v100, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s8, s14, -1
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v12, s18, v11
	v_dual_mov_b32 v101, 0 :: v_dual_add_nc_u32 v4, s16, v11
	v_dual_mov_b32 v106, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v102, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v104, 0 :: v_dual_add_nc_u32 v5, 64, v4
	v_min_i32_e32 v6, s8, v12
	v_min_i32_e32 v4, s3, v4
	v_min_i32_e32 v7, s8, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s3, v5
	v_dual_mov_b32 v154, 0 :: v_dual_and_b32 v15, 16, v15
	v_mad_co_u64_u32 v[68:69], null, 0x48, v6, v[3:4]
	v_mad_co_u64_u32 v[69:70], null, s2, v4, v[3:4]
	v_mad_co_u64_u32 v[70:71], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[71:72], null, s2, v5, v[3:4]
	s_wait_kmcnt 0x0
	global_load_b64 v[3:4], v68, s[6:7] offset:8
	global_load_b64 v[5:6], v69, s[4:5] offset:8
	global_load_b64 v[7:8], v70, s[6:7] offset:8
	global_load_b64 v[9:10], v71, s[4:5] offset:8
	v_lshrrev_b32_e32 v165, 5, v0
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_cmp_gt_i32_e64 s0, s14, v12
	v_mov_b32_e32 v126, 0
	v_or_b32_e32 v15, 8, v165
	v_and_or_b32 v16, v165, 6, v14
	v_lshlrev_b32_e32 v11, 3, v11
	v_cmp_gt_i32_e64 s1, s14, v13
	v_mov_b32_e32 v160, 0
	v_and_or_b32 v14, v15, 14, v14
	v_dual_mov_b32 v132, 0 :: v_dual_and_b32 v123, 15, v0
	v_lshl_or_b32 v15, v16, 8, v11
	v_mov_b32_e32 v121, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v11, v14, 8, v11
	v_bfe_u32 v164, v0, 4, 1
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v136, 0
	v_add_nc_u32_e32 v166, 0, v15
	v_add_nc_u32_e32 v167, 0, v11
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v156, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v163, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v6, 0x88888888, v6
	v_xor_b32_e32 v5, 0x88888888, v5
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v8, 0, v8, s1
	v_cndmask_b32_e64 v7, 0, v7, s1
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v10, 0x88888888, v10
	v_xor_b32_e32 v9, 0x88888888, v9
	ds_store_2addr_stride64_b64 v166, v[3:4], v[5:6] offset1:8
	ds_store_2addr_stride64_b64 v167, v[7:8], v[9:10] offset1:8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_14
; %bb.6:                                ; %.preheader519.lr.ph.i
	v_dual_mov_b32 v156, 0 :: v_dual_add_nc_u32 v3, s16, v2
	v_dual_mov_b32 v177, 0 :: v_dual_and_b32 v4, 31, v0
	v_lshrrev_b32_e32 v5, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_min_i32_e32 v7, s3, v3
	v_bfe_u32 v6, v0, 5, 1
	v_dual_mov_b32 v159, 0 :: v_dual_lshlrev_b32 v4, 3, v4
	v_dual_mov_b32 v163, 0 :: v_dual_add_nc_u32 v8, s18, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[65:66], null, s2, v7, s[4:5]
	v_ashrrev_i32_e32 v7, 31, v7
	v_dual_mov_b32 v176, 0 :: v_dual_lshlrev_b32 v9, 6, v164
	v_dual_mov_b32 v161, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	v_dual_mov_b32 v157, 0 :: v_dual_lshlrev_b32 v10, 2, v1
	v_dual_mov_b32 v162, 0 :: v_dual_lshlrev_b32 v11, 8, v5
	v_lshl_or_b32 v169, v6, 11, v4
	v_lshl_or_b32 v170, v5, 10, v4
	v_dual_mov_b32 v131, 0 :: v_dual_lshlrev_b32 v4, 9, v6
	v_dual_mov_b32 v160, 0 :: v_dual_lshlrev_b32 v5, 3, v123
	v_mad_co_u64_u32 v[66:67], null, s2, v7, v[66:67]
	v_min_i32_e32 v168, s8, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_add3_u32 v171, 0, v2, v10
	v_cmp_gt_i32_e64 s3, s12, v3
	v_dual_mov_b32 v129, 0 :: v_dual_lshlrev_b32 v172, 4, v1
	v_add3_u32 v173, 0, v11, v9
	v_add3_u32 v174, 0, v4, v5
	v_dual_mov_b32 v158, 0 :: v_dual_lshlrev_b32 v175, 2, v1
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v67, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v100, 0
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
.LBB3_8:                                ; %.preheader519.i
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
	s_wait_loadcnt_dscnt 0x205
	v_dual_mul_f32 v84, v98, v82 :: v_dual_mul_f32 v85, v99, v82
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v156, v84, v57 :: v_dual_fmac_f32 v163, v85, v58
	v_dual_mul_f32 v57, v96, v82 :: v_dual_mul_f32 v84, v97, v82
	v_mul_f32_e32 v58, v94, v82
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v50, v50
	v_dual_fmac_f32 v161, v57, v59 :: v_dual_fmac_f32 v162, v84, v60
	v_mul_f32_e32 v57, v95, v82
	v_cvt_f32_i32_e32 v59, v62
	v_dual_fmac_f32 v159, v58, v61 :: v_dual_mul_f32 v58, v88, v82
	v_mul_f32_e32 v60, v89, v82
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_fmac_f32_e32 v160, v57, v59
	v_mul_f32_e32 v57, v98, v83
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v157, v58, v61 :: v_dual_fmac_f32 v158, v60, v62
	v_mul_f32_e32 v58, v99, v83
	v_dual_fmac_f32 v154, v57, v49 :: v_dual_mul_f32 v49, v96, v83
	v_mul_f32_e32 v57, v97, v83
	v_cvt_f32_i32_e32 v52, v52
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v155, v58, v50
	v_mul_f32_e32 v50, v94, v83
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v41, v41
	v_fmac_f32_e32 v153, v57, v52
	v_fmac_f32_e32 v152, v49, v51
	v_mul_f32_e32 v49, v95, v83
	v_cvt_f32_i32_e32 v51, v54
	v_dual_fmac_f32 v151, v50, v53 :: v_dual_mul_f32 v50, v88, v83
	v_mul_f32_e32 v52, v89, v83
	v_cvt_f32_i32_e32 v53, v55
	s_wait_dscnt 0x4
	v_dual_fmac_f32 v150, v49, v51 :: v_dual_mul_f32 v49, v98, v72
	v_cvt_f32_i32_e32 v54, v56
	v_cvt_f32_i32_e32 v42, v42
	v_dual_fmac_f32 v149, v50, v53 :: v_dual_mul_f32 v50, v99, v72
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v146, v49, v41 :: v_dual_mul_f32 v41, v96, v72
	v_cvt_f32_i32_e32 v43, v43
	v_dual_fmac_f32 v148, v52, v54 :: v_dual_mul_f32 v49, v97, v72
	v_cvt_f32_i32_e32 v44, v44
	v_fmac_f32_e32 v147, v50, v42
	v_mul_f32_e32 v42, v94, v72
	v_cvt_f32_i32_e32 v45, v45
	v_dual_fmac_f32 v144, v41, v43 :: v_dual_mul_f32 v41, v95, v72
	v_cvt_f32_i32_e32 v43, v46
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v145, v49, v44 :: v_dual_fmac_f32 v142, v42, v45
	v_mul_f32_e32 v42, v88, v72
	v_cvt_f32_i32_e32 v45, v47
	v_fmac_f32_e32 v143, v41, v43
	v_mul_f32_e32 v41, v98, v73
	v_cvt_f32_i32_e32 v33, v33
	v_mul_f32_e32 v44, v89, v72
	v_cvt_f32_i32_e32 v46, v48
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v138, v41, v33
	v_mul_f32_e32 v33, v96, v73
	v_dual_fmac_f32 v140, v42, v45 :: v_dual_fmac_f32 v141, v44, v46
	v_dual_mul_f32 v42, v99, v73 :: v_dual_mul_f32 v41, v97, v73
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_dual_fmac_f32 v136, v33, v35 :: v_dual_mul_f32 v33, v95, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v139, v42, v34
	v_dual_mul_f32 v34, v94, v73 :: v_dual_fmac_f32 v137, v41, v36
	v_cvt_f32_i32_e32 v35, v38
	v_cvt_f32_i32_e32 v25, v25
	v_mul_f32_e32 v36, v89, v73
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v134, v34, v37
	v_mul_f32_e32 v34, v88, v73
	v_cvt_f32_i32_e32 v37, v39
	v_fmac_f32_e32 v135, v33, v35
	s_wait_dscnt 0x3
	v_mul_f32_e32 v33, v82, v80
	v_cvt_f32_i32_e32 v38, v40
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_wait_dscnt 0x2
	v_dual_fmac_f32 v130, v33, v25 :: v_dual_mul_f32 v25, v82, v78
	v_dual_fmac_f32 v132, v34, v37 :: v_dual_fmac_f32 v133, v36, v38
	v_dual_mul_f32 v34, v82, v81 :: v_dual_mul_f32 v33, v82, v79
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v131, v34, v26
	s_wait_dscnt 0x1
	v_mul_f32_e32 v26, v82, v76
	v_fmac_f32_e32 v128, v25, v27
	v_fmac_f32_e32 v129, v33, v28
	v_mul_f32_e32 v25, v82, v77
	v_cvt_f32_i32_e32 v27, v30
	v_fmac_f32_e32 v127, v26, v29
	s_wait_dscnt 0x0
	v_mul_f32_e32 v26, v82, v74
	v_cvt_f32_i32_e32 v29, v31
	v_mul_f32_e32 v28, v82, v75
	v_cvt_f32_i32_e32 v30, v32
	v_dual_fmac_f32 v126, v25, v27 :: v_dual_mul_f32 v25, v83, v80
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v124, v26, v29
	v_mul_f32_e32 v26, v83, v81
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_fmac_f32_e32 v121, v25, v17
	v_mul_f32_e32 v17, v83, v78
	v_dual_mul_f32 v25, v83, v79 :: v_dual_fmac_f32 v122, v26, v18
	v_mul_f32_e32 v18, v83, v76
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v120, v17, v19
	v_fmac_f32_e32 v119, v25, v20
	v_mul_f32_e32 v17, v83, v77
	v_cvt_f32_i32_e32 v19, v22
	v_dual_fmac_f32 v117, v18, v21 :: v_dual_mul_f32 v18, v83, v74
	v_cvt_f32_i32_e32 v21, v23
	v_mul_f32_e32 v20, v83, v75
	v_cvt_f32_i32_e32 v22, v24
	v_dual_fmac_f32 v118, v17, v19 :: v_dual_mul_f32 v17, v72, v80
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v115, v18, v21
	v_mul_f32_e32 v18, v72, v81
	v_cvt_f32_i32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v116, v20, v22 :: v_dual_fmac_f32 v113, v17, v9
	v_mul_f32_e32 v9, v72, v78
	v_dual_mul_f32 v17, v72, v79 :: v_dual_fmac_f32 v114, v18, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_mul_f32_e32 v10, v72, v76
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	v_fmac_f32_e32 v111, v9, v11
	v_dual_fmac_f32 v112, v17, v12 :: v_dual_mul_f32 v9, v72, v77
	v_cvt_f32_i32_e32 v11, v14
	v_dual_fmac_f32 v109, v10, v13 :: v_dual_mul_f32 v10, v72, v74
	v_mul_f32_e32 v12, v72, v75
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v73, v80 :: v_dual_mul_f32 v16, v73, v81
	v_cvt_f32_i32_e32 v2, v2
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v108, v12, v14 :: v_dual_fmac_f32 v105, v15, v1
	v_mul_f32_e32 v1, v73, v78
	v_fmac_f32_e32 v106, v16, v2
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v110, v9, v11 :: v_dual_fmac_f32 v107, v10, v13
	v_mul_f32_e32 v3, v73, v79
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v103, v1, v2
	v_dual_mul_f32 v1, v73, v76 :: v_dual_mul_f32 v2, v73, v77
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_dual_mul_f32 v9, v73, v74 :: v_dual_mul_f32 v10, v73, v75
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_fmac_f32 v125, v28, v30 :: v_dual_fmac_f32 v104, v3, v4
	v_dual_fmac_f32 v101, v1, v5 :: v_dual_fmac_f32 v102, v2, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_fmac_f32 v100, v9, v7 :: v_dual_fmac_f32 v67, v10, v8
	s_xor_b32 s8, s28, -1
	s_mov_b32 s30, 1
	s_mov_b32 s28, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s29, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_7
.LBB3_10:                               ;   Parent Loop BB3_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s34, s30, 6
	s_mov_b32 s35, s9
	s_or_b32 s8, s30, s11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[22:23], s[34:35]
	s_mul_u64 s[36:37], s[8:9], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, s31, s34, v69
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s35, 0, s31
	v_add_co_u32 v3, s31, s34, v71
	s_mul_u64 s[36:37], s[36:37], 0x48
	v_add_co_ci_u32_e64 v4, null, s35, 0, s31
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[6:7], s[36:37]
	s_clause 0x1
	global_load_b64 v[7:8], v[3:4], off offset:40
	global_load_b64 v[80:81], v[1:2], off offset:40
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s31, s36, v68
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s31
	v_add_co_u32 v1, s31, s36, v70
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, s37, 0, s31
	s_clause 0x1
	global_load_b64 v[92:93], v[5:6], off offset:40
	global_load_b64 v[84:85], v[1:2], off offset:40
	v_add_nc_u32_e32 v82, s26, v170
	v_add_nc_u32_e32 v83, s27, v169
	ds_load_2addr_stride64_b64 v[72:75], v82 offset1:1
	ds_load_2addr_stride64_b64 v[1:4], v83 offset1:1
	ds_load_2addr_stride64_b64 v[76:79], v83 offset0:2 offset1:3
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[3:4], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[76:77], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[78:79], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[1:2], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[3:4], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[76:77], 0 neg_lo:[1,1,0]
	s_wait_loadcnt 0x3
	v_xor_b32_e32 v87, 0x88888888, v8
	v_xor_b32_e32 v86, 0x88888888, v7
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v91, 0x88888888, v81
	v_xor_b32_e32 v90, 0x88888888, v80
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[78:79], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[72:75], v82 offset0:32 offset1:96
	ds_load_2addr_b64 v[76:79], v83 offset0:32 offset1:96
	ds_load_2addr_b64 v[80:83], v83 offset0:160 offset1:224
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v93, s0
	v_cndmask_b32_e64 v72, 0, v92, s0
	s_and_b32 s26, s29, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_2addr_stride64_b64 v166, v[90:91], v[72:73] offset0:16 offset1:32
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v85, s1
	v_cndmask_b32_e64 v72, 0, v84, s1
	ds_store_2addr_stride64_b64 v167, v[86:87], v[72:73] offset0:16 offset1:32
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.11:                               ;   in Loop: Header=BB3_10 Depth=2
	s_add_co_i32 s8, s8, 1
	s_xor_b32 s27, s30, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s30, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[30:31], s[8:9], 0x88
	s_mov_b32 s37, s9
	s_lshl_b32 s36, s27, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_add_nc_u64 s[30:31], s[4:5], s[30:31]
	s_mulk_i32 s8, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v72, s33, s34, v68
	s_add_nc_u64 s[30:31], s[30:31], s[36:37]
	v_mad_co_i64_i32 v[80:81], null, 0x48, v168, s[34:35]
	v_add_co_u32 v82, vcc_lo, v65, s8
	v_add_co_ci_u32_e64 v73, null, s35, 0, s33
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s30, v69
	v_add_co_u32 v76, s30, s30, v71
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s31, 0, s33
	s_lshl_b32 s8, s27, 2
	v_add_co_ci_u32_e64 v77, null, s31, 0, s30
	v_add_co_u32 v78, s30, s34, v70
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v82, vcc_lo, v82, s8
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v79, null, s35, 0, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v83, null, 0, v83, vcc_lo
	v_add_co_u32 v80, vcc_lo, v80, v175
	s_clause 0x1
	global_load_b64 v[74:75], v[74:75], off offset:8
	global_load_b64 v[76:77], v[76:77], off offset:8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v81, null, 0, v81, vcc_lo
	s_clause 0x1
	global_load_b64 v[92:93], v[72:73], off offset:8
	global_load_b64 v[84:85], v[78:79], off offset:8
	global_load_b32 v176, v[82:83], off
	global_load_b32 v177, v[80:81], off
	s_wait_loadcnt 0x5
	v_xor_b32_e32 v91, 0x88888888, v75
	s_wait_loadcnt 0x4
	v_xor_b32_e32 v87, 0x88888888, v77
	v_xor_b32_e32 v86, 0x88888888, v76
	v_xor_b32_e32 v90, 0x88888888, v74
.LBB3_12:                               ; %.preheader517.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	v_add_nc_u32_e32 v88, 0, v170
	v_add_nc_u32_e32 v89, 0, v169
	s_xor_b32 s8, s26, -1
	s_and_b32 s26, s28, exec_lo
	s_cselect_b32 s26, s13, 0x3400
	ds_load_2addr_stride64_b64 v[72:75], v88 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v89 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v89 offset0:34 offset1:35
	s_cselect_b32 s27, s19, 0x3c00
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v72, 0x100, v88
	v_add_nc_u32_e32 v80, 0x100, v89
	ds_load_2addr_stride64_b64 v[72:75], v72 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[76:79], v80 offset0:32 offset1:33
	ds_load_2addr_stride64_b64 v[80:83], v80 offset0:34 offset1:35
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[57:64], v[72:73], v[76:77], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[72:73], v[78:79], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[72:73], v[80:81], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[72:73], v[82:83], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[74:75], v[76:77], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[74:75], v[78:79], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[74:75], v[80:81], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[74:75], v[82:83], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v74, s27, v173
	v_add_nc_u32_e32 v72, s26, v174
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_movk_i32 s26, 0x2000
	s_movk_i32 s27, 0x4000
	ds_load_2addr_b32 v[98:99], v74 offset1:2
	ds_load_2addr_b32 v[96:97], v74 offset0:4 offset1:6
	ds_load_2addr_b32 v[94:95], v74 offset0:8 offset1:10
	ds_load_2addr_b32 v[88:89], v74 offset0:12 offset1:14
	ds_load_2addr_b32 v[82:83], v72 offset1:32
	ds_load_2addr_b32 v[72:73], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[80:81], v74 offset0:32 offset1:34
	ds_load_2addr_b32 v[78:79], v74 offset0:36 offset1:38
	ds_load_2addr_b32 v[76:77], v74 offset0:40 offset1:42
	ds_load_2addr_b32 v[74:75], v74 offset0:44 offset1:46
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_9
; %bb.13:                               ; %.preheader518.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	s_wait_loadcnt 0x1
	v_lshrrev_b32_e32 v178, v172, v176
	s_and_b32 s8, s29, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	v_cndmask_b32_e64 v93, 0, v93, s0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v180, s8, v171
	v_cvt_f32_f16_e64 v178, v178.l
	s_cselect_b32 s8, s19, 0x3c00
	v_cndmask_b32_e64 v92, 0, v92, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v179, 0, v177, s2
	v_cndmask_b32_e64 v85, 0, v85, s1
	v_cndmask_b32_e64 v84, 0, v84, s1
	v_cndmask_b32_e64 v178, 0, v178, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v181, s8, v171
	s_mov_b32 s27, 0
	s_movk_i32 s26, 0x1000
	ds_store_2addr_stride64_b64 v166, v[92:93], v[90:91] offset1:8
	ds_store_2addr_stride64_b64 v167, v[84:85], v[86:87] offset1:8
	ds_store_b32 v180, v179
	ds_store_b32 v181, v178
	s_branch .LBB3_9
.LBB3_14:                               ; %._crit_edge585.i
	v_mul_u32_u24_e32 v1, 0x500, v165
	v_lshlrev_b32_e32 v2, 2, v123
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
	v_mul_u32_u24_e32 v4, 0x50, v123
	s_cselect_b32 s1, -1, 0
	v_lshlrev_b32_e32 v5, 2, v3
	v_mad_u32_u24 v2, 0x280, v164, v7
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s0, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_105
; %bb.15:                               ; %.preheader514.i
	ds_store_2addr_b32 v2, v156, v163 offset1:20
	ds_store_2addr_b32 v2, v161, v162 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v159, v160 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v157, v158 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_or_b32_e32 v0, s16, v123
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
	s_cbranch_execz .LBB3_17
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
.LBB3_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_23
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
.LBB3_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_27
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
.LBB3_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v10, 0x280, v164
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_31
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
.LBB3_31:                               ; %.preheader512.1.i
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
	ds_store_2addr_b32 v7, v154, v155 offset1:20
	ds_store_2addr_b32 v7, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v151, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v149, v148 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_33
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
.LBB3_41:                               ; %.preheader512.2.i
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
	ds_store_2addr_b32 v7, v146, v147 offset1:20
	ds_store_2addr_b32 v7, v144, v145 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v142, v143 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v140, v141 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_43
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
.LBB3_51:                               ; %.preheader512.3.i
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
	ds_store_2addr_b32 v7, v138, v139 offset1:20
	ds_store_2addr_b32 v7, v136, v137 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v134, v135 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v132, v133 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB3_53
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
.LBB3_61:                               ; %.preheader513.1.i
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
	ds_store_2addr_b32 v7, v130, v131 offset1:20
	ds_store_2addr_b32 v7, v128, v129 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v127, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v124, v125 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_63
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
.LBB3_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_65
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
.LBB3_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_69
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
.LBB3_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_73
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
.LBB3_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_77
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
.LBB3_77:                               ; %.preheader512.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s11, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v121, v122 offset1:20
	ds_store_2addr_b32 v7, v120, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v115, v116 offset0:120 offset1:140
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
.LBB3_86:                               ; %.preheader512.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v113, v114 offset1:20
	ds_store_2addr_b32 v7, v111, v112 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v109, v110 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v107, v108 offset0:120 offset1:140
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
.LBB3_95:                               ; %.preheader512.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_and_b32 s1, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v7, v105, v106 offset1:20
	ds_store_2addr_b32 v7, v103, v104 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v101, v102 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v100, v67 offset0:120 offset1:140
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
.LBB3_104:                              ; %.loopexit.loopexit618.i
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
; %bb.106:                              ; %.preheader511.i
	ds_store_2addr_b32 v2, v156, v163 offset1:20
	ds_store_2addr_b32 v2, v161, v162 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v159, v160 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v157, v158 offset0:120 offset1:140
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
	v_add_lshl_u32 v11, v3, v123, 2
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
	ds_store_2addr_b32 v2, v154, v155 offset1:20
	ds_store_2addr_b32 v2, v152, v153 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v151, v150 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v149, v148 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v146, v147 offset1:20
	ds_store_2addr_b32 v2, v144, v145 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v142, v143 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v140, v141 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v138, v139 offset1:20
	ds_store_2addr_b32 v2, v136, v137 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v134, v135 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v132, v133 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v130, v131 offset1:20
	ds_store_2addr_b32 v2, v128, v129 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v127, v126 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v124, v125 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v121, v122 offset1:20
	ds_store_2addr_b32 v2, v120, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v117, v118 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v115, v116 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v113, v114 offset1:20
	ds_store_2addr_b32 v2, v111, v112 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v109, v110 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v107, v108 offset0:120 offset1:140
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
	ds_store_2addr_b32 v2, v105, v106 offset1:20
	ds_store_2addr_b32 v2, v103, v104 offset0:40 offset1:60
	ds_store_2addr_b32 v2, v101, v102 offset0:80 offset1:100
	ds_store_2addr_b32 v2, v100, v67 offset0:120 offset1:140
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
.LBB3_107:                              ; %.loopexit.i
	s_wait_loadcnt 0x0
	global_inv scope:SCOPE_SE
.LBB3_108:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB3_109:
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
	s_cbranch_execz .LBB3_35
.LBB3_110:
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
	s_cbranch_execz .LBB3_36
.LBB3_111:
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
	s_cbranch_execz .LBB3_37
.LBB3_112:
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
	s_cbranch_execz .LBB3_38
.LBB3_113:
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
	s_cbranch_execz .LBB3_39
.LBB3_114:
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
	s_cbranch_execnz .LBB3_40
	s_branch .LBB3_41
.LBB3_115:
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
	s_cbranch_execz .LBB3_45
.LBB3_116:
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
	s_cbranch_execz .LBB3_46
.LBB3_117:
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
	s_cbranch_execz .LBB3_47
.LBB3_118:
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
	s_cbranch_execz .LBB3_48
.LBB3_119:
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
	s_cbranch_execz .LBB3_49
.LBB3_120:
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
	s_cbranch_execnz .LBB3_50
	s_branch .LBB3_51
.LBB3_121:
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
	s_cbranch_execz .LBB3_55
.LBB3_122:
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
	s_cbranch_execz .LBB3_56
.LBB3_123:
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
	s_cbranch_execz .LBB3_57
.LBB3_124:
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
	s_cbranch_execz .LBB3_58
.LBB3_125:
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
	s_cbranch_execz .LBB3_59
.LBB3_126:
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
	s_cbranch_execnz .LBB3_60
	s_branch .LBB3_61
.LBB3_127:
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
	s_cbranch_execz .LBB3_79
.LBB3_128:
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
	s_cbranch_execz .LBB3_80
.LBB3_129:
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
	s_cbranch_execz .LBB3_81
.LBB3_130:
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
	s_cbranch_execz .LBB3_82
.LBB3_131:
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
	s_cbranch_execz .LBB3_83
.LBB3_132:
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
	s_cbranch_execz .LBB3_84
.LBB3_133:
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
	s_cbranch_execnz .LBB3_85
	s_branch .LBB3_86
.LBB3_134:
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
	s_cbranch_execz .LBB3_88
.LBB3_135:
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
	s_cbranch_execz .LBB3_89
.LBB3_136:
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
	s_cbranch_execz .LBB3_90
.LBB3_137:
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
	s_cbranch_execz .LBB3_91
.LBB3_138:
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
	s_cbranch_execz .LBB3_92
.LBB3_139:
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
	s_cbranch_execz .LBB3_93
.LBB3_140:
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
	s_cbranch_execnz .LBB3_94
	s_branch .LBB3_95
.LBB3_141:
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
	s_cbranch_execz .LBB3_97
.LBB3_142:
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
	s_cbranch_execz .LBB3_98
.LBB3_143:
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
	s_cbranch_execz .LBB3_99
.LBB3_144:
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
	s_cbranch_execz .LBB3_100
.LBB3_145:
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
	s_cbranch_execz .LBB3_101
.LBB3_146:
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
	s_cbranch_execz .LBB3_102
.LBB3_147:
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
	s_cbranch_execnz .LBB3_103
	s_branch .LBB3_104
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
		.amdhsa_next_free_vgpr 182
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_vgpr, 182
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
; codeLenInByte = 15400
; TotalNumSgprs: 40
; NumVgprs: 182
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 182
; Occupancy: 8
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
	.type	__hip_cuid_a8e8abf1211b8b64,@object ; @__hip_cuid_a8e8abf1211b8b64
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a8e8abf1211b8b64
__hip_cuid_a8e8abf1211b8b64:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a8e8abf1211b8b64, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a8e8abf1211b8b64
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
    .vgpr_count:     182
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
    .vgpr_count:     182
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
    .vgpr_count:     182
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
