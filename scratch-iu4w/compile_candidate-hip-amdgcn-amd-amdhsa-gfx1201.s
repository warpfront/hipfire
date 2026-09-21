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
	s_load_b128 s[8:11], s[0:1], 0x18
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x10
	s_lshl_b32 s20, ttmp9, 7
	s_lshl_b32 s16, ttmp7, 7
	v_lshrrev_b32_e32 v89, 5, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s20, s8
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s16, s10
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s24, s0, s1
	s_cmp_eq_u32 s11, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB1_12
; %bb.1:
	s_mov_b32 s25, 0
	s_and_b32 vcc_lo, exec_lo, s24
	s_cbranch_vccz .LBB1_13
; %bb.2:                                ; %.preheader427.i
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_10
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v6, 1, v0
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v5, 1, v0
	v_mov_b32_e32 v7, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v1, s16, v6
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[7:8], null, 0x48, v1, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v7, vcc_lo, v7, v3
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	global_load_b32 v7, v[7:8], off
.LBB1_5:                                ; %.preheader424.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, s20, v6
	s_add_co_i32 s1, s8, -1
	s_ashr_i32 s3, s8, 31
	s_mov_b32 s2, s8
	s_ashr_i32 s5, s9, 31
	v_min_i32_e32 v8, s1, v4
	s_mov_b32 s4, s9
	v_lshlrev_b32_e32 v6, 3, v6
	s_mul_u64 s[2:3], s[4:5], s[2:3]
	s_mov_b32 s1, exec_lo
	v_ashrrev_i32_e32 v1, 4, v8
	s_lshr_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v8, 15, v8
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	v_add3_u32 v6, 0, v6, v3
	v_lshlrev_b64_e32 v[1:2], 6, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v7 offset:4096
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v9, vcc_lo
                                        ; implicit-def: $vgpr1_lo16
	v_cmpx_ne_u32_e32 0, v5
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB1_7
; %bb.6:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off offset:48
	global_load_d16_hi_u8 v1, v[2:3], off offset:32
                                        ; implicit-def: $vgpr2_vgpr3
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB1_7:                                ; %Flow1989
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB1_9
; %bb.8:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off
	global_load_d16_hi_u8 v1, v[2:3], off offset:16
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB1_9:                                ; %._crit_edge.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cmp_gt_i32_e32 vcc_lo, s8, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	ds_store_b32 v6, v1 offset:6144
.LBB1_10:                               ; %Flow1990
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v5, 2, v0
	v_and_b32_e32 v1, 3, v0
	s_add_co_i32 s3, s10, -1
	v_lshlrev_b32_e32 v8, 4, v0
	v_bfe_u32 v9, v0, 1, 1
	v_add_nc_u32_e32 v6, s16, v5
	v_lshlrev_b32_e32 v1, 3, v1
	s_mov_b32 s2, -1
	v_and_b32_e32 v8, 16, v8
	v_and_or_b32 v10, v89, 6, v9
	v_add_nc_u32_e32 v7, 64, v6
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v2, s3, v6
	v_cmp_gt_i32_e64 s0, s10, v6
	v_and_or_b32 v5, v5, 15, v8
	v_or_b32_e32 v8, 8, v89
	v_min_i32_e32 v3, s3, v7
	v_mad_co_u64_u32 v[65:66], null, 0x48, v2, v[1:2]
	v_cmp_gt_i32_e64 s1, s10, v7
	v_lshlrev_b32_e32 v5, 3, v5
	v_and_or_b32 v8, v8, 14, v9
	v_mad_co_u64_u32 v[66:67], null, 0x48, v3, v[1:2]
	s_cmp_gt_i32 s9, 0xff
	s_clause 0x1
	global_load_b64 v[1:2], v65, s[14:15] offset:8
	global_load_b64 v[3:4], v66, s[14:15] offset:8
	v_lshl_or_b32 v9, v10, 8, v5
	v_lshl_or_b32 v5, v8, 8, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v148, 0, v9
	v_add_nc_u32_e32 v149, 0, v5
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v4, 0, v4, s1
	v_cndmask_b32_e64 v3, 0, v3, s1
	ds_store_b64 v148, v[1:2]
	ds_store_b64 v149, v[3:4]
	v_bfe_u32 v1, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v147, 3, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_14
; %bb.11:                               ; %._crit_edge.._crit_edge479_crit_edge.i
	v_lshlrev_b32_e32 v14, 3, v1
	s_ashr_i32 s23, s8, 31
	s_mov_b32 s22, s8
	s_mov_b32 s2, 0
	s_branch .LBB1_15
.LBB1_12:
	s_mov_b32 s25, -1
.LBB1_13:                               ; %Flow2129
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB1_410
	s_branch .LBB1_816
.LBB1_14:
                                        ; implicit-def: $sgpr22_sgpr23
                                        ; implicit-def: $vgpr14
.LBB1_15:                               ; %Flow1986
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v93, 0 :: v_dual_and_b32 v146, 15, v0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v157, 0
	v_mov_b32_e32 v156, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_25
; %bb.16:                               ; %.preheader423.lr.ph.i
	v_dual_mov_b32 v68, 0 :: v_dual_lshlrev_b32 v3, 4, v89
	s_add_co_i32 s5, s8, -16
	s_ashr_i32 s4, s9, 6
	v_and_b32_e32 v4, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v170, v68 :: v_dual_add_nc_u32 v1, s20, v3
	v_lshrrev_b32_e32 v5, 1, v0
	s_lshl_b32 s26, s4, 9
	v_mov_b32_e32 v169, v68
	v_cmp_gt_i32_e64 s2, s8, v1
	v_dual_mov_b32 v155, v68 :: v_dual_lshlrev_b32 v158, 3, v4
	v_dual_mov_b32 v153, v68 :: v_dual_lshlrev_b32 v4, 4, v4
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v1, s5, v1, s2
	s_ashr_i32 s5, s4, 31
	v_dual_mov_b32 v151, v68 :: v_dual_add_nc_u32 v10, s20, v5
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[4:5], s[4:5], 23
	v_ashrrev_i32_e32 v1, 4, v1
	s_add_co_i32 s21, s8, -1
	s_ashr_i32 s23, s8, 31
	s_mov_b32 s22, s8
	s_mov_b32 s7, 0
	v_ashrrev_i32_e32 v7, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v8, s4, v1
	v_mad_co_u64_u32 v[1:2], null, s26, v1, s[12:13]
	s_mov_b32 s6, s9
	v_dual_mov_b32 v157, v68 :: v_dual_and_b32 v6, 1, v0
	v_mul_lo_u32 v7, s26, v7
	s_mul_u64 s[26:27], s[6:7], s[22:23]
	v_dual_mov_b32 v156, v68 :: v_dual_add_nc_u32 v9, s16, v5
	v_add_co_u32 v160, vcc_lo, v1, v4
	v_min_i32_e32 v1, s21, v10
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[26:27], s[26:27], 1
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v2, v8, v2, v7
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[26:27], s[12:13], s[26:27]
	s_ashr_i32 s17, s9, 31
	v_min_i32_e32 v159, s3, v9
	v_ashrrev_i32_e32 v67, 4, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, 0, v2, vcc_lo
	v_dual_mov_b32 v121, v68 :: v_dual_and_b32 v2, 15, v1
	v_dual_mov_b32 v154, v68 :: v_dual_lshlrev_b32 v1, 3, v5
	v_dual_mov_b32 v119, v68 :: v_dual_lshlrev_b32 v4, 2, v6
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v162, s3, s26, v2
	v_or_b32_e32 v2, v147, v3
	s_lshr_b32 s5, s17, 24
	v_add_co_ci_u32_e64 v163, null, s27, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s9, s5
	v_cmp_gt_i32_e64 s3, s10, v9
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 8
	v_add3_u32 v164, 0, v1, v4
	v_cmp_gt_i32_e64 s4, s8, v10
	v_dual_mov_b32 v152, v68 :: v_dual_lshlrev_b32 v165, 4, v6
	v_lshl_add_u32 v166, v2, 3, 0
	v_lshl_add_u32 v167, v146, 3, 0
	v_dual_mov_b32 v117, v68 :: v_dual_lshlrev_b32 v168, 2, v6
	v_dual_mov_b32 v150, v68 :: v_dual_mov_b32 v115, v68
	v_dual_mov_b32 v120, v68 :: v_dual_mov_b32 v145, v68
	v_dual_mov_b32 v118, v68 :: v_dual_mov_b32 v143, v68
	v_dual_mov_b32 v116, v68 :: v_dual_mov_b32 v141, v68
	v_dual_mov_b32 v114, v68 :: v_dual_mov_b32 v139, v68
	v_dual_mov_b32 v144, v68 :: v_dual_mov_b32 v113, v68
	v_dual_mov_b32 v142, v68 :: v_dual_mov_b32 v111, v68
	v_dual_mov_b32 v140, v68 :: v_dual_mov_b32 v109, v68
	v_dual_mov_b32 v138, v68 :: v_dual_mov_b32 v107, v68
	v_dual_mov_b32 v112, v68 :: v_dual_mov_b32 v137, v68
	v_dual_mov_b32 v110, v68 :: v_dual_mov_b32 v135, v68
	v_dual_mov_b32 v108, v68 :: v_dual_mov_b32 v133, v68
	v_dual_mov_b32 v106, v68 :: v_dual_mov_b32 v131, v68
	v_dual_mov_b32 v136, v68 :: v_dual_mov_b32 v105, v68
	v_dual_mov_b32 v134, v68 :: v_dual_mov_b32 v103, v68
	v_dual_mov_b32 v132, v68 :: v_dual_mov_b32 v101, v68
	v_dual_mov_b32 v130, v68 :: v_dual_mov_b32 v99, v68
	v_dual_mov_b32 v104, v68 :: v_dual_mov_b32 v129, v68
	v_dual_mov_b32 v102, v68 :: v_dual_mov_b32 v127, v68
	v_dual_mov_b32 v100, v68 :: v_dual_mov_b32 v125, v68
	v_dual_mov_b32 v98, v68 :: v_dual_mov_b32 v123, v68
	v_dual_mov_b32 v128, v68 :: v_dual_mov_b32 v97, v68
	v_dual_mov_b32 v126, v68 :: v_dual_mov_b32 v95, v68
	v_dual_mov_b32 v124, v68 :: v_dual_mov_b32 v93, v68
	v_dual_mov_b32 v122, v68 :: v_dual_mov_b32 v91, v68
	v_mov_b32_e32 v96, v68
	v_mov_b32_e32 v94, v68
	v_mov_b32_e32 v92, v68
	v_mov_b32_e32 v90, v68
	s_ashr_i32 s17, s8, 4
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s21, s17, 31
	s_movk_i32 s26, 0x1000
	s_movk_i32 s27, 0x1800
	s_mov_b32 s31, 0
	s_mov_b32 s28, 0
	s_branch .LBB1_18
.LBB1_17:                               ;   in Loop: Header=BB1_18 Depth=1
	s_and_b32 vcc_lo, exec_lo, s30
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_24
.LBB1_18:                               ; %.preheader423.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_20 Depth 2
	s_lshl_b32 s29, s28, 1
	s_add_co_i32 s28, s28, 1
	s_mov_b32 s34, s7
	s_cmp_eq_u32 s28, s5
	s_mov_b32 s33, -1
	s_cselect_b32 s30, -1, 0
	s_mov_b32 s35, s7
	s_branch .LBB1_20
.LBB1_19:                               ;   in Loop: Header=BB1_20 Depth=2
	s_wait_dscnt 0x3
	v_dual_mul_f32 v79, v77, v87 :: v_dual_mul_f32 v80, v78, v87
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_dual_fmac_f32 v156, v79, v57 :: v_dual_mul_f32 v79, v76, v87
	v_dual_fmac_f32 v157, v80, v58 :: v_dual_mul_f32 v58, v71, v87
	v_mul_f32_e32 v57, v75, v87
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v155, v79, v60 :: v_dual_mul_f32 v60, v70, v87
	v_fmac_f32_e32 v152, v58, v61
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v57, v59
	v_dual_mul_f32 v57, v72, v87 :: v_dual_mul_f32 v58, v69, v87
	v_cvt_f32_i32_e32 v59, v62
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v153, v57, v59 :: v_dual_fmac_f32 v150, v58, v61
	v_mul_f32_e32 v57, v77, v88
	v_dual_fmac_f32 v151, v60, v62 :: v_dual_mul_f32 v58, v78, v88
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v53, v53
	v_dual_fmac_f32 v144, v57, v49 :: v_dual_mul_f32 v49, v75, v88
	v_mul_f32_e32 v57, v76, v88
	v_dual_fmac_f32 v145, v58, v50 :: v_dual_mul_f32 v50, v71, v88
	v_cvt_f32_i32_e32 v41, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v142, v49, v51
	v_fmac_f32_e32 v143, v57, v52
	v_mul_f32_e32 v49, v72, v88
	v_cvt_f32_i32_e32 v51, v54
	v_fmac_f32_e32 v140, v50, v53
	v_mul_f32_e32 v50, v69, v88
	v_cvt_f32_i32_e32 v53, v55
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v52, v70, v88 :: v_dual_fmac_f32 v141, v49, v51
	v_cvt_f32_i32_e32 v54, v56
	s_wait_dscnt 0x2
	v_mul_f32_e32 v49, v77, v85
	v_fmac_f32_e32 v138, v50, v53
	v_mul_f32_e32 v50, v78, v85
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v43, v43
	v_fmac_f32_e32 v136, v49, v41
	v_mul_f32_e32 v41, v75, v85
	v_mul_f32_e32 v49, v76, v85
	v_cvt_f32_i32_e32 v44, v44
	v_dual_fmac_f32 v137, v50, v42 :: v_dual_mul_f32 v42, v71, v85
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v135, v49, v44
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v132, v42, v45
	v_mul_f32_e32 v42, v69, v85
	v_dual_fmac_f32 v134, v41, v43 :: v_dual_mul_f32 v41, v72, v85
	v_cvt_f32_i32_e32 v43, v46
	v_cvt_f32_i32_e32 v45, v47
	v_mul_f32_e32 v44, v70, v85
	v_cvt_f32_i32_e32 v46, v48
	v_cvt_f32_i32_e32 v36, v36
	v_fmac_f32_e32 v133, v41, v43
	v_dual_mul_f32 v41, v77, v86 :: v_dual_fmac_f32 v130, v42, v45
	v_mul_f32_e32 v42, v78, v86
	v_fmac_f32_e32 v131, v44, v46
	v_cvt_f32_i32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v128, v41, v33 :: v_dual_mul_f32 v33, v75, v86
	v_mul_f32_e32 v41, v76, v86
	v_fmac_f32_e32 v129, v42, v34
	v_mul_f32_e32 v34, v71, v86
	v_cvt_f32_i32_e32 v25, v25
	v_fmac_f32_e32 v126, v33, v35
	v_fmac_f32_e32 v127, v41, v36
	v_mul_f32_e32 v33, v72, v86
	v_cvt_f32_i32_e32 v35, v38
	v_fmac_f32_e32 v124, v34, v37
	v_mul_f32_e32 v34, v69, v86
	v_mul_f32_e32 v36, v70, v86
	v_cvt_f32_i32_e32 v37, v39
	v_cvt_f32_i32_e32 v38, v40
	v_fmac_f32_e32 v125, v33, v35
	s_wait_dscnt 0x1
	v_mul_f32_e32 v33, v77, v83
	v_cvt_f32_i32_e32 v26, v26
	v_dual_fmac_f32 v122, v34, v37 :: v_dual_fmac_f32 v123, v36, v38
	v_mul_f32_e32 v34, v78, v83
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v120, v33, v25 :: v_dual_mul_f32 v25, v75, v83
	v_mul_f32_e32 v33, v76, v83
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_fmac_f32 v121, v34, v26 :: v_dual_mul_f32 v26, v71, v83
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v118, v25, v27
	v_fmac_f32_e32 v119, v33, v28
	v_mul_f32_e32 v25, v72, v83
	v_cvt_f32_i32_e32 v27, v30
	v_fmac_f32_e32 v116, v26, v29
	v_mul_f32_e32 v26, v69, v83
	v_mul_f32_e32 v28, v70, v83
	v_cvt_f32_i32_e32 v29, v31
	v_cvt_f32_i32_e32 v30, v32
	v_fmac_f32_e32 v117, v25, v27
	v_mul_f32_e32 v25, v77, v84
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v114, v26, v29 :: v_dual_fmac_f32 v115, v28, v30
	v_mul_f32_e32 v26, v78, v84
	v_cvt_f32_i32_e32 v18, v18
	v_dual_fmac_f32 v112, v25, v17 :: v_dual_mul_f32 v17, v75, v84
	v_mul_f32_e32 v25, v76, v84
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_dual_fmac_f32 v113, v26, v18 :: v_dual_mul_f32 v18, v71, v84
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v110, v17, v19
	v_fmac_f32_e32 v111, v25, v20
	v_mul_f32_e32 v17, v72, v84
	v_cvt_f32_i32_e32 v19, v22
	v_fmac_f32_e32 v108, v18, v21
	v_mul_f32_e32 v18, v69, v84
	v_cvt_f32_i32_e32 v21, v23
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v20, v70, v84 :: v_dual_fmac_f32 v109, v17, v19
	v_cvt_f32_i32_e32 v22, v24
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v77, v73
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v106, v18, v21
	v_mul_f32_e32 v18, v78, v73
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v104, v17, v9
	v_mul_f32_e32 v9, v75, v73
	v_mul_f32_e32 v17, v76, v73
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v105, v18, v10 :: v_dual_mul_f32 v10, v71, v73
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v103, v17, v12 :: v_dual_mul_f32 v12, v70, v73
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v100, v10, v13
	v_mul_f32_e32 v10, v69, v73
	v_dual_fmac_f32 v102, v9, v11 :: v_dual_mul_f32 v9, v72, v73
	v_cvt_f32_i32_e32 v11, v14
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v77, v74 :: v_dual_mul_f32 v16, v78, v74
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v101, v9, v11
	s_barrier_signal -1
	v_dual_fmac_f32 v96, v15, v1 :: v_dual_fmac_f32 v97, v16, v2
	v_mul_f32_e32 v1, v75, v74
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v98, v10, v13 :: v_dual_fmac_f32 v99, v12, v14
	v_dual_mul_f32 v3, v76, v74 :: v_dual_mul_f32 v10, v70, v74
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v1, v2
	v_mul_f32_e32 v2, v72, v74
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_mul_f32_e32 v9, v69, v74
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v1, v71, v74
	v_fmac_f32_e32 v139, v52, v54
	v_fmac_f32_e32 v107, v20, v22
	v_dual_fmac_f32 v95, v3, v4 :: v_dual_fmac_f32 v92, v2, v6
	v_dual_fmac_f32 v91, v9, v7 :: v_dual_fmac_f32 v90, v10, v8
	v_fmac_f32_e32 v93, v1, v5
	s_xor_b32 s6, s33, -1
	s_mov_b32 s35, 1
	s_mov_b32 s33, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s34, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_20:                               ;   Parent Loop BB1_18 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s35, s29
	s_mov_b32 s37, s7
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s36, s6, 1
	v_add_nc_u32_e32 v83, s31, v158
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[36:37], s[36:37], 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v160, s36
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s37, v161, vcc_lo
	s_mul_u64 s[36:37], s[6:7], s[10:11]
	ds_load_2addr_stride64_b64 v[71:74], v83 offset0:6 offset1:7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	global_load_b128 v[1:4], v[69:70], off
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[14:15], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s35, s36, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s35
	v_add_co_u32 v7, s35, s36, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s37, 0, s35
	s_clause 0x1
	global_load_b64 v[79:80], v[5:6], off offset:40
	global_load_b64 v[81:82], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[5:8], v83 offset1:1
	ds_load_2addr_stride64_b64 v[9:12], v83 offset0:2 offset1:3
	ds_load_2addr_stride64_b64 v[13:16], v83 offset0:4 offset1:5
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v2, 0x88888888, v2
	v_xor_b32_e32 v1, 0x88888888, v1
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v76, 0, v2, s2
	v_cndmask_b32_e64 v75, 0, v1, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v78, 0, v4, s2
	v_cndmask_b32_e64 v77, 0, v3, s2
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[75:76], v[5:6], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[75:76], v[7:8], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[75:76], v[9:10], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[75:76], v[11:12], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[75:76], v[13:14], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[75:76], v[15:16], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[75:76], v[71:72], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[75:76], v[73:74], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[71:74], v83 offset0:32 offset1:96
	v_add_nc_u32_e32 v75, 0x100, v83
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[71:72], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[73:74], v[49:56] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[71:74], v83 offset0:160 offset1:224
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[71:72], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[73:74], v[33:40] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[71:72], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[73:74], v[17:24] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:6 offset1:7
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[71:72], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[73:74], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v80, s0
	v_cndmask_b32_e64 v71, 0, v79, s0
	s_and_b32 s31, s34, s30
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s31
	ds_store_b64 v148, v[71:72] offset:8192
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v82, s1
	v_cndmask_b32_e64 v71, 0, v81, s1
	ds_store_b64 v149, v[71:72] offset:8192
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_22
; %bb.21:                               ;   in Loop: Header=BB1_20 Depth=2
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[71:72], null, s6, s17, v[67:68]
	s_mul_u64 s[36:37], s[6:7], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[14:15], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[75:76], null, 0x48, v159, s[36:37]
	v_mad_co_u64_u32 v[72:73], null, s6, s21, v[72:73]
	v_add_co_u32 v73, s6, s36, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, s37, 0, s6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[71:72], 6, v[71:72]
	v_add_co_u32 v71, vcc_lo, v162, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, v163, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v75, v168
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v76, vcc_lo
	s_clause 0x3
	global_load_u8 v77, v[71:72], off offset:16
	global_load_u8 v78, v[71:72], off offset:32
	global_load_u8 v83, v[71:72], off offset:48
	global_load_u8 v84, v[71:72], off
	v_add_co_u32 v71, s6, s36, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, s37, 0, s6
	s_clause 0x2
	global_load_b64 v[79:80], v[73:74], off offset:8
	global_load_b64 v[81:82], v[71:72], off offset:8
	global_load_b32 v169, v[75:76], off
	s_wait_loadcnt 0x5
	v_lshlrev_b32_e32 v72, 16, v78
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v73, 24, v83
	s_wait_loadcnt 0x3
	v_lshl_or_b32 v71, v84, 8, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_or3_b32 v170, v71, v72, v73
.LBB1_22:                               ; %.preheader421.i
                                        ;   in Loop: Header=BB1_20 Depth=2
	global_load_b128 v[69:72], v[69:70], off offset:512
	v_add_nc_u32_e32 v87, 0, v158
	s_xor_b32 s6, s31, -1
	s_and_b32 s31, s33, exec_lo
	s_cselect_b32 s31, s26, 0x1400
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[175:178], v87 offset0:22 offset1:23
	s_cselect_b32 s35, s27, 0x1c00
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v70, 0x88888888, v70
	v_xor_b32_e32 v69, 0x88888888, v69
	v_xor_b32_e32 v71, 0x88888888, v71
	v_xor_b32_e32 v72, 0x88888888, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v70, 0, v70, s2
	v_cndmask_b32_e64 v69, 0, v69, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v77, 0, v71, s2
	v_cndmask_b32_e64 v78, 0, v72, s2
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[69:70], v[73:74], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[69:70], v[75:76], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[69:70], v[83:84], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[69:70], v[85:86], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[69:70], v[171:172], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[69:70], v[173:174], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[69:70], v[175:176], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[69:70], v[177:178], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v87, 0x100, v87
	ds_load_2addr_stride64_b64 v[69:72], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:22 offset1:23
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[71:72], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[73:74], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[75:76], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[83:84], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[85:86], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[173:174], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v69, s35, v166
	v_add_nc_u32_e32 v73, s31, v167
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_movk_i32 s31, 0x2000
	ds_load_2addr_b32 v[77:78], v69 offset1:2
	ds_load_2addr_b32 v[75:76], v69 offset0:4 offset1:6
	ds_load_2addr_b32 v[71:72], v69 offset0:8 offset1:10
	ds_load_2addr_b32 v[69:70], v69 offset0:12 offset1:14
	ds_load_2addr_b32 v[87:88], v73 offset1:32
	ds_load_2addr_b32 v[85:86], v73 offset0:64 offset1:96
	ds_load_2addr_b32 v[83:84], v73 offset0:128 offset1:160
	ds_load_2addr_b32 v[73:74], v73 offset0:192 offset1:224
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_19
; %bb.23:                               ; %.preheader422.i
                                        ;   in Loop: Header=BB1_20 Depth=2
	v_lshrrev_b32_e32 v171, v165, v170
	s_and_b32 s6, s34, exec_lo
	s_cselect_b32 s6, s26, 0x1400
	v_cndmask_b32_e64 v80, 0, v80, s0
	v_cndmask_b32_e64 v79, 0, v79, s0
	v_cvt_f32_f16_e64 v171, v171.l
	v_cndmask_b32_e64 v82, 0, v82, s1
	v_cndmask_b32_e64 v81, 0, v81, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v173, s6, v164
	s_cselect_b32 s6, s27, 0x1c00
	v_cndmask_b32_e64 v172, 0, v169, s3
	v_cndmask_b32_e64 v171, 0, v171, s4
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v174, s6, v164
	s_mov_b32 s31, 0
	ds_store_b64 v148, v[79:80]
	ds_store_b64 v149, v[81:82]
	ds_store_b32 v173, v172
	ds_store_b32 v174, v171
	s_branch .LBB1_19
.LBB1_24:                               ; %Flow1984
	v_mov_b32_e32 v14, v147
.LBB1_25:                               ; %._crit_edge479.i
	v_mul_u32_u24_e32 v1, 0x500, v89
	v_lshlrev_b32_e32 v2, 2, v146
	v_lshrrev_b32_e32 v6, 7, v0
	v_and_b32_e32 v4, 0x7f, v0
	s_ashr_i32 s17, s16, 31
	s_ashr_i32 s21, s20, 31
	v_add3_u32 v15, 0, v1, v2
	v_mul_u32_u24_e32 v2, 0x50, v146
	v_lshlrev_b32_e32 v5, 2, v6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[22:23], s[16:17]
	s_lshl_b64 s[2:3], s[20:21], 2
	v_mad_i32_i24 v1, 0x50, v14, v15
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_co_i32 s6, s20, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[18:19], s[0:1]
	ds_store_2addr_b32 v1, v156, v157 offset1:20
	ds_store_2addr_b32 v1, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v152, v153 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v150, v151 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_bfe_u32 v1, v0, 4, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mul_lo_u32 v7, s8, v6
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s6, s8
	v_mad_u32_u24 v3, 0x500, v1, 0
	v_or_b32_e32 v1, s20, v4
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s16, 0x80
	s_mov_b32 s7, 0x31004000
	v_add3_u32 v3, v3, v2, v5
	v_ashrrev_i32_e32 v2, 31, v1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s10
	s_mov_b32 s6, -1
	s_cselect_b32 s1, -1, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v5, v3
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, s1
	v_cmp_gt_i32_e64 s0, s8, v1
	s_mov_b32 s2, -1
	v_add_co_u32 v1, vcc_lo, s18, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, s19, v9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_29
; %bb.26:
	v_or_b32_e32 v8, s16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v8
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_28
; %bb.27:
	v_ashrrev_i32_e32 v10, 31, v8
	v_mul_lo_u32 v11, s23, v8
	v_mad_co_u64_u32 v[8:9], null, s22, v8, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v10, s22, v10
	v_add3_u32 v9, v9, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v2, v9, vcc_lo
	global_load_b32 v10, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v5, v10
	global_store_b32 v[8:9], v10, off
.LBB1_28:                               ; %Flow1980
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_29:                               ; %Flow1981
	v_add_lshl_u32 v4, v7, v4, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_31
; %bb.30:
	buffer_load_b32 v7, v4, s[4:7], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v7
	buffer_store_b32 v5, v4, s[4:7], null offen
.LBB1_31:
	ds_load_b32 v8, v3 offset:8
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v5, 0, 1, s1
	v_or_b32_e32 v7, 2, v6
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_35
; %bb.32:
	v_or_b32_e32 v9, s16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v9
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_34
; %bb.33:
	v_ashrrev_i32_e32 v11, 31, v9
	v_mul_lo_u32 v12, s23, v9
	v_mad_co_u64_u32 v[9:10], null, s22, v9, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, s22, v11
	v_add3_u32 v10, v10, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, v1, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v2, v10, vcc_lo
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB1_34:                               ; %Flow1978
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_35:                               ; %Flow1979
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_37
; %bb.36:
	s_lshl_b32 s1, s8, 3
	buffer_load_b32 v9, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v4, s[4:7], s1 offen
.LBB1_37:
	ds_load_b32 v9, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v8, 4, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_41
; %bb.38:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v10, s16, v8
	v_cmp_gt_i32_e32 vcc_lo, s10, v10
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_40
; %bb.39:
	v_ashrrev_i32_e32 v12, 31, v10
	v_mul_lo_u32 v13, s23, v10
	v_mad_co_u64_u32 v[10:11], null, s22, v10, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s22, v12
	v_add3_u32 v11, v11, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v2, v11, vcc_lo
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v9, v12
	global_store_b32 v[10:11], v12, off
.LBB1_40:                               ; %Flow1976
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_41:                               ; %Flow1977
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_43
; %bb.42:
	s_lshl_b32 s1, s8, 4
	buffer_load_b32 v10, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	buffer_store_b32 v9, v4, s[4:7], s1 offen
.LBB1_43:
	ds_load_b32 v10, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 6, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_47
; %bb.44:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v11, s16, v9
	v_cmp_gt_i32_e32 vcc_lo, s10, v11
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_46
; %bb.45:
	v_ashrrev_i32_e32 v13, 31, v11
	v_mul_lo_u32 v16, s23, v11
	v_mad_co_u64_u32 v[11:12], null, s22, v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v13, s22, v13
	v_add3_u32 v12, v12, v13, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, vcc_lo, v1, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v2, v12, vcc_lo
	global_load_b32 v13, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v10, v13
	global_store_b32 v[11:12], v13, off
.LBB1_46:                               ; %Flow1974
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_47:                               ; %Flow1975
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_49
; %bb.48:
	s_mul_i32 s1, s8, 24
	buffer_load_b32 v11, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	buffer_store_b32 v10, v4, s[4:7], s1 offen
.LBB1_49:
	ds_load_b32 v11, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v10, 8, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_53
; %bb.50:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, s16, v10
	v_cmp_gt_i32_e32 vcc_lo, s10, v12
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_52
; %bb.51:
	v_ashrrev_i32_e32 v16, 31, v12
	v_mul_lo_u32 v17, s23, v12
	v_mad_co_u64_u32 v[12:13], null, s22, v12, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v16, s22, v16
	v_add3_u32 v13, v13, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v12, vcc_lo, v1, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v2, v13, vcc_lo
	global_load_b32 v16, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v11, v16
	global_store_b32 v[12:13], v16, off
.LBB1_52:                               ; %Flow1972
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_53:                               ; %Flow1973
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_55
; %bb.54:
	s_lshl_b32 s1, s8, 5
	buffer_load_b32 v12, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v11, v12
	buffer_store_b32 v11, v4, s[4:7], s1 offen
.LBB1_55:
	ds_load_b32 v12, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 10, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_59
; %bb.56:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v13, s16, v11
	v_cmp_gt_i32_e32 vcc_lo, s10, v13
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_58
; %bb.57:
	v_ashrrev_i32_e32 v18, 31, v13
	v_mul_lo_u32 v19, s23, v13
	v_mad_co_u64_u32 v[16:17], null, s22, v13, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v13, s22, v18
	v_add3_u32 v17, v17, v13, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v13, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v12, v13
	global_store_b32 v[16:17], v13, off
.LBB1_58:                               ; %Flow1970
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_59:                               ; %Flow1971
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_61
; %bb.60:
	s_mul_i32 s1, s8, 40
	buffer_load_b32 v13, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v12, v13
	buffer_store_b32 v12, v4, s[4:7], s1 offen
.LBB1_61:
	ds_load_b32 v13, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v12, 12, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_65
; %bb.62:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v16, s16, v12
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_64
; %bb.63:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v13, v18
	global_store_b32 v[16:17], v18, off
.LBB1_64:                               ; %Flow1968
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_65:                               ; %Flow1969
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_67
; %bb.66:
	s_mul_i32 s1, s8, 48
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v13, v16
	buffer_store_b32 v13, v4, s[4:7], s1 offen
.LBB1_67:
	ds_load_b32 v16, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_wait_dscnt 0x1
	v_or_b32_e32 v13, 14, v6
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_71
; %bb.68:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v17, s16, v13
	v_cmp_gt_i32_e32 vcc_lo, s10, v17
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_70
; %bb.69:
	v_ashrrev_i32_e32 v19, 31, v17
	v_mul_lo_u32 v20, s23, v17
	v_mad_co_u64_u32 v[17:18], null, s22, v17, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v19, s22, v19
	v_add3_u32 v18, v18, v19, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[17:18], 2, v[17:18]
	v_add_co_u32 v17, vcc_lo, v1, v17
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v18, null, v2, v18, vcc_lo
	global_load_b32 v19, v[17:18], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v19, v16, v19
	global_store_b32 v[17:18], v19, off
.LBB1_70:                               ; %Flow1966
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_71:                               ; %Flow1967
	v_mul_i32_i24_e32 v14, 0x50, v14
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_73
; %bb.72:
	s_mul_i32 s1, s8, 56
	buffer_load_b32 v17, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v16, v16, v17
	buffer_store_b32 v16, v4, s[4:7], s1 offen
.LBB1_73:                               ; %.preheader.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v14, v15, v14
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 16
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v144, v145 offset1:20
	ds_store_2addr_b32 v14, v142, v143 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v140, v141 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v138, v139 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_77
; %bb.74:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_76
; %bb.75:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_76:                               ; %Flow1964
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_77:                               ; %Flow1965
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_79
; %bb.78:
	s_lshl_b32 s2, s8, 6
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_79:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_83
; %bb.80:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_82
; %bb.81:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_82:                               ; %Flow1962
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_83:                               ; %Flow1963
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_85
; %bb.84:
	s_mul_i32 s2, s8, 0x48
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_85:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_89
; %bb.86:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_88
; %bb.87:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_88:                               ; %Flow1960
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_89:                               ; %Flow1961
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_91
; %bb.90:
	s_mul_i32 s2, s8, 0x50
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_91:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_95
; %bb.92:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_94
; %bb.93:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_94:                               ; %Flow1958
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_95:                               ; %Flow1959
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_97
; %bb.96:
	s_mul_i32 s2, s8, 0x58
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_97:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_101
; %bb.98:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_100
; %bb.99:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_100:                              ; %Flow1956
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_101:                              ; %Flow1957
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_103
; %bb.102:
	s_mul_i32 s2, s8, 0x60
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_103:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_107
; %bb.104:
	v_add_nc_u32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_106
; %bb.105:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_106:                              ; %Flow1954
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_107:                              ; %Flow1955
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_109
; %bb.108:
	s_mul_i32 s2, s8, 0x68
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_109:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_113
; %bb.110:
	v_add_nc_u32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_112
; %bb.111:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_112:                              ; %Flow1952
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_113:                              ; %Flow1953
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_115
; %bb.114:
	s_mul_i32 s2, s8, 0x70
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_115:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_119
; %bb.116:
	v_add_nc_u32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_118
; %bb.117:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_118:                              ; %Flow1950
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_119:                              ; %Flow1951
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_121
; %bb.120:
	s_mul_i32 s1, s8, 0x78
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_121:                              ; %.preheader.2.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 32
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v136, v137 offset1:20
	ds_store_2addr_b32 v14, v134, v135 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v132, v133 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v130, v131 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_125
; %bb.122:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_124
; %bb.123:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_124:                              ; %Flow1948
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_125:                              ; %Flow1949
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_127
; %bb.126:
	s_lshl_b32 s2, s8, 7
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_127:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_131
; %bb.128:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_130
; %bb.129:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_130:                              ; %Flow1946
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_131:                              ; %Flow1947
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_133
; %bb.132:
	s_mul_i32 s2, s8, 0x88
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_133:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_137
; %bb.134:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_136
; %bb.135:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_136:                              ; %Flow1944
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_137:                              ; %Flow1945
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_139
; %bb.138:
	s_mul_i32 s2, s8, 0x90
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_139:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_143
; %bb.140:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_142
; %bb.141:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_142:                              ; %Flow1942
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_143:                              ; %Flow1943
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_145
; %bb.144:
	s_mul_i32 s2, s8, 0x98
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_145:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_149
; %bb.146:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_148
; %bb.147:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_148:                              ; %Flow1940
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_149:                              ; %Flow1941
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_151
; %bb.150:
	s_mul_i32 s2, s8, 0xa0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_151:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_155
; %bb.152:
	v_or_b32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_154
; %bb.153:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_154:                              ; %Flow1938
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_155:                              ; %Flow1939
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_157
; %bb.156:
	s_mul_i32 s2, s8, 0xa8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_157:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_161
; %bb.158:
	v_or_b32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_160
; %bb.159:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_160:                              ; %Flow1936
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_161:                              ; %Flow1937
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_163
; %bb.162:
	s_mul_i32 s2, s8, 0xb0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_163:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_167
; %bb.164:
	v_or_b32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_166
; %bb.165:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_166:                              ; %Flow1934
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_167:                              ; %Flow1935
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_169
; %bb.168:
	s_mul_i32 s1, s8, 0xb8
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_169:                              ; %.preheader.3.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 48
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v128, v129 offset1:20
	ds_store_2addr_b32 v14, v126, v127 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v124, v125 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v122, v123 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_173
; %bb.170:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_172
; %bb.171:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_172:                              ; %Flow1932
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_173:                              ; %Flow1933
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_175
; %bb.174:
	s_mul_i32 s2, s8, 0xc0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_175:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_179
; %bb.176:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_178
; %bb.177:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_178:                              ; %Flow1930
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_179:                              ; %Flow1931
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_181
; %bb.180:
	s_mul_i32 s2, s8, 0xc8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_181:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_185
; %bb.182:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_184
; %bb.183:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_184:                              ; %Flow1928
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_185:                              ; %Flow1929
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_187
; %bb.186:
	s_mul_i32 s2, s8, 0xd0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_187:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_191
; %bb.188:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_190
; %bb.189:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_190:                              ; %Flow1926
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_191:                              ; %Flow1927
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_193
; %bb.192:
	s_mul_i32 s2, s8, 0xd8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_193:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_197
; %bb.194:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_196
; %bb.195:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_196:                              ; %Flow1924
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_197:                              ; %Flow1925
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_199
; %bb.198:
	s_mul_i32 s2, s8, 0xe0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_199:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_203
; %bb.200:
	v_add_nc_u32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_202
; %bb.201:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_202:                              ; %Flow1922
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_203:                              ; %Flow1923
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_205
; %bb.204:
	s_mul_i32 s2, s8, 0xe8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_205:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_209
; %bb.206:
	v_add_nc_u32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_208
; %bb.207:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_208:                              ; %Flow1920
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_209:                              ; %Flow1921
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_211
; %bb.210:
	s_mul_i32 s2, s8, 0xf0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_211:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_215
; %bb.212:
	v_add_nc_u32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_214
; %bb.213:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_214:                              ; %Flow1918
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_215:                              ; %Flow1919
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_217
; %bb.216:
	s_mul_i32 s1, s8, 0xf8
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_217:                              ; %.preheader419.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 64
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v120, v121 offset1:20
	ds_store_2addr_b32 v14, v118, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v116, v117 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v114, v115 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_221
; %bb.218:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_220
; %bb.219:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_220:                              ; %Flow1916
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_221:                              ; %Flow1917
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_223
; %bb.222:
	s_lshl_b32 s2, s8, 8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_223:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_227
; %bb.224:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_226
; %bb.225:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_226:                              ; %Flow1914
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_227:                              ; %Flow1915
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_229
; %bb.228:
	s_mul_i32 s2, s8, 0x108
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_229:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_233
; %bb.230:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_232
; %bb.231:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_232:                              ; %Flow1912
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_233:                              ; %Flow1913
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_235
; %bb.234:
	s_mul_i32 s2, s8, 0x110
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_235:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_239
; %bb.236:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_238
; %bb.237:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_238:                              ; %Flow1910
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_239:                              ; %Flow1911
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_241
; %bb.240:
	s_mul_i32 s2, s8, 0x118
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_241:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_245
; %bb.242:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_244
; %bb.243:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_244:                              ; %Flow1908
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_245:                              ; %Flow1909
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_247
; %bb.246:
	s_mul_i32 s2, s8, 0x120
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_247:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_251
; %bb.248:
	v_or_b32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_250
; %bb.249:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_250:                              ; %Flow1906
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_251:                              ; %Flow1907
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_253
; %bb.252:
	s_mul_i32 s2, s8, 0x128
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_253:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_257
; %bb.254:
	v_or_b32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_256
; %bb.255:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_256:                              ; %Flow1904
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_257:                              ; %Flow1905
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_259
; %bb.258:
	s_mul_i32 s2, s8, 0x130
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_259:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_263
; %bb.260:
	v_or_b32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_262
; %bb.261:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_262:                              ; %Flow1902
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_263:                              ; %Flow1903
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_265
; %bb.264:
	s_mul_i32 s1, s8, 0x138
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_265:                              ; %.preheader.1.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 0x50
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v112, v113 offset1:20
	ds_store_2addr_b32 v14, v110, v111 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v108, v109 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v106, v107 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_269
; %bb.266:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_268
; %bb.267:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_268:                              ; %Flow1900
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_269:                              ; %Flow1901
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_271
; %bb.270:
	s_mul_i32 s2, s8, 0x140
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_271:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_275
; %bb.272:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_274
; %bb.273:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_274:                              ; %Flow1898
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_275:                              ; %Flow1899
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_277
; %bb.276:
	s_mul_i32 s2, s8, 0x148
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_277:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_281
; %bb.278:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_280
; %bb.279:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_280:                              ; %Flow1896
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_281:                              ; %Flow1897
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_283
; %bb.282:
	s_mul_i32 s2, s8, 0x150
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_283:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_287
; %bb.284:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_286
; %bb.285:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_286:                              ; %Flow1894
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_287:                              ; %Flow1895
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_289
; %bb.288:
	s_mul_i32 s2, s8, 0x158
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_289:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_293
; %bb.290:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_292
; %bb.291:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_292:                              ; %Flow1892
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_293:                              ; %Flow1893
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_295
; %bb.294:
	s_mul_i32 s2, s8, 0x160
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_295:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_299
; %bb.296:
	v_add_nc_u32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_298
; %bb.297:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_298:                              ; %Flow1890
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_299:                              ; %Flow1891
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_301
; %bb.300:
	s_mul_i32 s2, s8, 0x168
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_301:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_305
; %bb.302:
	v_add_nc_u32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_304
; %bb.303:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_304:                              ; %Flow1888
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_305:                              ; %Flow1889
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_307
; %bb.306:
	s_mul_i32 s2, s8, 0x170
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_307:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_311
; %bb.308:
	v_add_nc_u32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_310
; %bb.309:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_310:                              ; %Flow1886
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_311:                              ; %Flow1887
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_313
; %bb.312:
	s_mul_i32 s1, s8, 0x178
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_313:                              ; %.preheader.2.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 0x60
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v104, v105 offset1:20
	ds_store_2addr_b32 v14, v102, v103 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v100, v101 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v98, v99 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v15, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_317
; %bb.314:
	v_or_b32_e32 v16, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_316
; %bb.315:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_316:                              ; %Flow1884
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_317:                              ; %Flow1885
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_319
; %bb.318:
	s_mul_i32 s2, s8, 0x180
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_319:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_323
; %bb.320:
	v_or_b32_e32 v16, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_322
; %bb.321:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_322:                              ; %Flow1882
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_323:                              ; %Flow1883
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_325
; %bb.324:
	s_mul_i32 s2, s8, 0x188
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_325:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_329
; %bb.326:
	v_or_b32_e32 v16, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_328
; %bb.327:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_328:                              ; %Flow1880
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_329:                              ; %Flow1881
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_331
; %bb.330:
	s_mul_i32 s2, s8, 0x190
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_331:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_335
; %bb.332:
	v_or_b32_e32 v16, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_334
; %bb.333:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_334:                              ; %Flow1878
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_335:                              ; %Flow1879
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_337
; %bb.336:
	s_mul_i32 s2, s8, 0x198
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_337:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_341
; %bb.338:
	v_or_b32_e32 v16, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_340
; %bb.339:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_340:                              ; %Flow1876
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_341:                              ; %Flow1877
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_343
; %bb.342:
	s_mul_i32 s2, s8, 0x1a0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_343:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_347
; %bb.344:
	v_or_b32_e32 v16, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_346
; %bb.345:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_346:                              ; %Flow1874
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_347:                              ; %Flow1875
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_349
; %bb.348:
	s_mul_i32 s2, s8, 0x1a8
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_349:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_353
; %bb.350:
	v_or_b32_e32 v16, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_352
; %bb.351:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_352:                              ; %Flow1872
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_353:                              ; %Flow1873
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_355
; %bb.354:
	s_mul_i32 s2, s8, 0x1b0
	buffer_load_b32 v16, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s2 offen
.LBB1_355:
	s_wait_dscnt 0x0
	ds_load_b32 v15, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_359
; %bb.356:
	v_or_b32_e32 v16, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_358
; %bb.357:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s23, v16
	v_mad_co_u64_u32 v[16:17], null, s22, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s22, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v1, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v2, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB1_358:                              ; %Flow1870
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_359:                              ; %Flow1871
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_361
; %bb.360:
	s_mul_i32 s1, s8, 0x1b8
	buffer_load_b32 v16, v4, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v4, s[4:7], s1 offen
.LBB1_361:                              ; %.preheader.3.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_or_b32 s1, s16, 0x70
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v14, v96, v97 offset1:20
	ds_store_2addr_b32 v14, v94, v95 offset0:40 offset1:60
	ds_store_2addr_b32 v14, v93, v92 offset0:80 offset1:100
	ds_store_2addr_b32 v14, v91, v90 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v3
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_365
; %bb.362:
	v_or_b32_e32 v6, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_364
; %bb.363:
	v_ashrrev_i32_e32 v17, 31, v6
	v_mul_lo_u32 v18, s23, v6
	v_mad_co_u64_u32 v[15:16], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s22, v17
	v_add3_u32 v16, v16, v6, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v1, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v2, v16, vcc_lo
	global_load_b32 v6, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v14, v6
	global_store_b32 v[15:16], v6, off
.LBB1_364:                              ; %Flow1868
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_365:                              ; %Flow1869
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_367
; %bb.366:
	s_mul_i32 s2, s8, 0x1c0
	buffer_load_b32 v6, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v14, v6
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_367:
	ds_load_b32 v6, v3 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_371
; %bb.368:
	v_or_b32_e32 v7, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_370
; %bb.369:
	v_ashrrev_i32_e32 v16, 31, v7
	v_mul_lo_u32 v17, s23, v7
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[14:15], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v7, s22, v16
	v_add3_u32 v15, v15, v7, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v1, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v2, v15, vcc_lo
	global_load_b32 v7, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v6, v7
	global_store_b32 v[14:15], v7, off
.LBB1_370:                              ; %Flow1866
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_371:                              ; %Flow1867
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_373
; %bb.372:
	s_mul_i32 s2, s8, 0x1c8
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_373:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_377
; %bb.374:
	v_or_b32_e32 v7, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_376
; %bb.375:
	v_ashrrev_i32_e32 v14, 31, v7
	v_mul_lo_u32 v15, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v14, s22, v14
	v_add3_u32 v8, v8, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v1, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v2, v8, vcc_lo
	global_load_b32 v14, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v6, v14
	global_store_b32 v[7:8], v14, off
.LBB1_376:                              ; %Flow1864
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_377:                              ; %Flow1865
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_379
; %bb.378:
	s_mul_i32 s2, s8, 0x1d0
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_379:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_383
; %bb.380:
	v_or_b32_e32 v7, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_382
; %bb.381:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v14, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s22, v9
	v_add3_u32 v8, v8, v9, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v1, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v2, v8, vcc_lo
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_382:                              ; %Flow1862
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_383:                              ; %Flow1863
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_385
; %bb.384:
	s_mul_i32 s2, s8, 0x1d8
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_385:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_389
; %bb.386:
	v_or_b32_e32 v7, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_388
; %bb.387:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s22, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v1, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v2, v8, vcc_lo
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_388:                              ; %Flow1860
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_389:                              ; %Flow1861
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_391
; %bb.390:
	s_mul_i32 s2, s8, 0x1e0
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_391:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_395
; %bb.392:
	v_add_nc_u32_e32 v7, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_394
; %bb.393:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s22, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v1, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v2, v8, vcc_lo
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_394:                              ; %Flow1858
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_395:                              ; %Flow1859
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_397
; %bb.396:
	s_mul_i32 s2, s8, 0x1e8
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_397:
	s_wait_dscnt 0x0
	ds_load_b32 v6, v3 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_401
; %bb.398:
	v_add_nc_u32_e32 v7, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_400
; %bb.399:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s22, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v1, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v2, v8, vcc_lo
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v6, v9
	global_store_b32 v[7:8], v9, off
.LBB1_400:                              ; %Flow1856
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_401:                              ; %Flow1857
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_403
; %bb.402:
	s_mul_i32 s2, s8, 0x1f0
	buffer_load_b32 v7, v4, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v6, v7
	buffer_store_b32 v6, v4, s[4:7], s2 offen
.LBB1_403:
	ds_load_b32 v3, v3 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v5
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_407
; %bb.404:
	v_add_nc_u32_e32 v5, s1, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v5
	s_and_b32 s1, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_406
; %bb.405:
	v_ashrrev_i32_e32 v7, 31, v5
	v_mul_lo_u32 v8, s23, v5
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[5:6], null, s22, v5, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v7, s22, v7
	v_add3_u32 v6, v6, v7, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v1, vcc_lo, v1, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, v2, v6, vcc_lo
	global_load_b32 v5, v[1:2], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v3, v5
	global_store_b32 v[1:2], v5, off
.LBB1_406:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB1_407:                              ; %Flow1855
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_409
; %bb.408:
	s_mul_i32 s0, s8, 0x1f8
	buffer_load_b32 v1, v4, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v1, v3, v1
	buffer_store_b32 v1, v4, s[4:7], s0 offen
.LBB1_409:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
	s_and_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_816
.LBB1_410:
	s_and_b32 vcc_lo, exec_lo, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_816
; %bb.411:                              ; %.preheader422.i17
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB1_419
; %bb.412:                              ; %.lr.ph.i130
	v_lshrrev_b32_e32 v6, 1, v0
	v_dual_mov_b32 v2, 0 :: v_dual_and_b32 v5, 1, v0
	v_mov_b32_e32 v7, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_or_b32_e32 v1, s16, v6
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB1_414
; %bb.413:
	v_mad_co_i64_i32 v[7:8], null, 0x48, v1, s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v7, vcc_lo, v7, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	global_load_b32 v7, v[7:8], off
.LBB1_414:                              ; %.preheader419.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v4, s20, v6
	s_add_co_i32 s1, s8, -1
	s_ashr_i32 s3, s8, 31
	s_mov_b32 s2, s8
	s_ashr_i32 s5, s9, 31
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v8, s1, v4
	s_mov_b32 s4, s9
	v_lshlrev_b32_e32 v6, 3, v6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[2:3], s[4:5], s[2:3]
	s_mov_b32 s1, exec_lo
	v_ashrrev_i32_e32 v1, 4, v8
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v8, 15, v8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[12:13], s[2:3]
	v_add3_u32 v6, 0, v6, v3
	v_lshlrev_b64_e32 v[1:2], 6, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b32 v6, v7 offset:4096
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v9, null, s3, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v1, v8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v9, vcc_lo
                                        ; implicit-def: $vgpr1_lo16
	v_cmpx_ne_u32_e32 0, v5
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB1_416
; %bb.415:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off offset:48
	global_load_d16_hi_u8 v1, v[2:3], off offset:32
                                        ; implicit-def: $vgpr2_vgpr3
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB1_416:                              ; %Flow2127
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB1_418
; %bb.417:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off
	global_load_d16_hi_u8 v1, v[2:3], off offset:16
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB1_418:                              ; %._crit_edge.loopexit.i131
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cmp_gt_i32_e32 vcc_lo, s8, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	ds_store_b32 v6, v1 offset:6144
.LBB1_419:                              ; %Flow2128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v5, 2, v0
	v_and_b32_e32 v1, 3, v0
	s_add_co_i32 s3, s10, -1
	v_lshlrev_b32_e32 v8, 4, v0
	v_bfe_u32 v9, v0, 1, 1
	v_add_nc_u32_e32 v6, s16, v5
	v_lshlrev_b32_e32 v1, 3, v1
	s_mov_b32 s2, -1
	v_and_b32_e32 v8, 16, v8
	v_and_or_b32 v10, v89, 6, v9
	v_add_nc_u32_e32 v7, 64, v6
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v2, s3, v6
	v_cmp_gt_i32_e64 s0, s10, v6
	v_and_or_b32 v5, v5, 15, v8
	v_or_b32_e32 v8, 8, v89
	v_min_i32_e32 v3, s3, v7
	v_mad_co_u64_u32 v[65:66], null, 0x48, v2, v[1:2]
	v_cmp_gt_i32_e64 s1, s10, v7
	v_lshlrev_b32_e32 v5, 3, v5
	v_and_or_b32 v8, v8, 14, v9
	v_mad_co_u64_u32 v[66:67], null, 0x48, v3, v[1:2]
	s_cmp_gt_i32 s9, 0xff
	s_clause 0x1
	global_load_b64 v[1:2], v65, s[14:15] offset:8
	global_load_b64 v[3:4], v66, s[14:15] offset:8
	v_lshl_or_b32 v9, v10, 8, v5
	v_lshl_or_b32 v5, v8, 8, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v148, 0, v9
	v_add_nc_u32_e32 v149, 0, v5
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v4, 0, v4, s1
	v_cndmask_b32_e64 v3, 0, v3, s1
	ds_store_b64 v148, v[1:2]
	ds_store_b64 v149, v[3:4]
	v_bfe_u32 v1, v0, 4, 1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_421
; %bb.420:                              ; %._crit_edge.._crit_edge474_crit_edge.i
	v_lshlrev_b32_e32 v138, 3, v1
	s_ashr_i32 s23, s8, 31
	s_mov_b32 s22, s8
	s_mov_b32 s2, 0
	s_branch .LBB1_422
.LBB1_421:
                                        ; implicit-def: $sgpr22_sgpr23
                                        ; implicit-def: $vgpr138
.LBB1_422:                              ; %Flow2124
	v_dual_mov_b32 v90, 0 :: v_dual_and_b32 v147, 15, v0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v96, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v156, 0
	v_mov_b32_e32 v157, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_431
; %bb.423:                              ; %.preheader418.lr.ph.i
	v_dual_mov_b32 v68, 0 :: v_dual_and_b32 v5, 31, v0
	v_lshlrev_b32_e32 v4, 4, v89
	s_add_co_i32 s5, s8, -16
	s_ashr_i32 s4, s9, 6
	v_lshrrev_b32_e32 v6, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v169, v68 :: v_dual_add_nc_u32 v2, s20, v4
	v_mov_b32_e32 v157, v68
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s24, s4, 9
	v_dual_mov_b32 v155, v68 :: v_dual_lshlrev_b32 v158, 3, v5
	v_cmp_gt_i32_e64 s2, s8, v2
	v_dual_mov_b32 v156, v68 :: v_dual_lshlrev_b32 v5, 4, v5
	v_dual_mov_b32 v154, v68 :: v_dual_add_nc_u32 v11, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v2, s5, v2, s2
	s_ashr_i32 s5, s4, 31
	s_add_co_i32 s21, s8, -1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[4:5], s[4:5], 23
	s_ashr_i32 s23, s8, 31
	v_ashrrev_i32_e32 v2, 4, v2
	s_mov_b32 s22, s8
	s_mov_b32 s7, 0
	s_mov_b32 s6, s9
	v_dual_mov_b32 v170, v68 :: v_dual_and_b32 v7, 1, v0
	v_ashrrev_i32_e32 v8, 31, v2
	s_wait_alu depctr_sa_sdst(0)
	v_mul_lo_u32 v9, s4, v2
	v_mad_co_u64_u32 v[2:3], null, s24, v2, s[12:13]
	v_dual_mov_b32 v153, v68 :: v_dual_add_nc_u32 v10, s16, v6
	v_mul_lo_u32 v8, s24, v8
	s_mul_u64 s[24:25], s[6:7], s[22:23]
	v_dual_mov_b32 v121, v68 :: v_dual_lshlrev_b32 v138, 3, v1
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[24:25], s[24:25], 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_u32 v160, vcc_lo, v2, v5
	v_min_i32_e32 v2, s21, v11
	v_add3_u32 v3, v9, v3, v8
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[24:25], s[12:13], s[24:25]
	s_ashr_i32 s17, s9, 31
	v_min_i32_e32 v159, s3, v10
	v_ashrrev_i32_e32 v67, 4, v2
	v_dual_mov_b32 v151, v68 :: v_dual_and_b32 v2, 15, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, 0, v3, vcc_lo
	v_dual_mov_b32 v152, v68 :: v_dual_lshlrev_b32 v1, 3, v6
	v_dual_mov_b32 v150, v68 :: v_dual_lshlrev_b32 v3, 2, v7
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v162, s3, s24, v2
	v_or_b32_e32 v2, v138, v4
	s_lshr_b32 s5, s17, 24
	v_add_co_ci_u32_e64 v163, null, s25, 0, s3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s9, s5
	v_cmp_gt_i32_e64 s3, s10, v10
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s5, s4, 8
	v_add3_u32 v164, 0, v1, v3
	v_cmp_gt_i32_e64 s4, s8, v11
	v_dual_mov_b32 v120, v68 :: v_dual_lshlrev_b32 v165, 4, v7
	v_lshl_add_u32 v166, v2, 3, 0
	v_lshl_add_u32 v167, v147, 3, 0
	v_dual_mov_b32 v119, v68 :: v_dual_lshlrev_b32 v168, 2, v7
	v_dual_mov_b32 v118, v68 :: v_dual_mov_b32 v117, v68
	v_dual_mov_b32 v116, v68 :: v_dual_mov_b32 v115, v68
	v_dual_mov_b32 v114, v68 :: v_dual_mov_b32 v145, v68
	v_dual_mov_b32 v146, v68 :: v_dual_mov_b32 v143, v68
	v_dual_mov_b32 v144, v68 :: v_dual_mov_b32 v141, v68
	v_dual_mov_b32 v142, v68 :: v_dual_mov_b32 v139, v68
	v_dual_mov_b32 v140, v68 :: v_dual_mov_b32 v113, v68
	v_dual_mov_b32 v112, v68 :: v_dual_mov_b32 v111, v68
	v_dual_mov_b32 v110, v68 :: v_dual_mov_b32 v109, v68
	v_dual_mov_b32 v108, v68 :: v_dual_mov_b32 v107, v68
	v_dual_mov_b32 v106, v68 :: v_dual_mov_b32 v137, v68
	v_dual_mov_b32 v136, v68 :: v_dual_mov_b32 v135, v68
	v_dual_mov_b32 v134, v68 :: v_dual_mov_b32 v133, v68
	v_dual_mov_b32 v132, v68 :: v_dual_mov_b32 v131, v68
	v_dual_mov_b32 v130, v68 :: v_dual_mov_b32 v105, v68
	v_dual_mov_b32 v104, v68 :: v_dual_mov_b32 v103, v68
	v_dual_mov_b32 v102, v68 :: v_dual_mov_b32 v101, v68
	v_dual_mov_b32 v100, v68 :: v_dual_mov_b32 v99, v68
	v_dual_mov_b32 v98, v68 :: v_dual_mov_b32 v129, v68
	v_dual_mov_b32 v128, v68 :: v_dual_mov_b32 v127, v68
	v_dual_mov_b32 v126, v68 :: v_dual_mov_b32 v125, v68
	v_dual_mov_b32 v124, v68 :: v_dual_mov_b32 v123, v68
	v_dual_mov_b32 v122, v68 :: v_dual_mov_b32 v97, v68
	v_dual_mov_b32 v96, v68 :: v_dual_mov_b32 v95, v68
	v_dual_mov_b32 v94, v68 :: v_dual_mov_b32 v93, v68
	v_dual_mov_b32 v92, v68 :: v_dual_mov_b32 v91, v68
	v_mov_b32_e32 v90, v68
	s_ashr_i32 s9, s8, 4
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s12, s9, 31
	s_movk_i32 s13, 0x1000
	s_movk_i32 s17, 0x1800
	s_mov_b32 s26, 0
	s_mov_b32 s21, 0
	s_branch .LBB1_425
.LBB1_424:                              ;   in Loop: Header=BB1_425 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_431
.LBB1_425:                              ; %.preheader418.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_427 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s24, s21, 1
	s_add_co_i32 s21, s21, 1
	s_mov_b32 s28, s7
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s21, s5
	s_mov_b32 s27, -1
	s_cselect_b32 s25, -1, 0
	s_mov_b32 s29, s7
	s_branch .LBB1_427
.LBB1_426:                              ;   in Loop: Header=BB1_427 Depth=2
	s_wait_dscnt 0x3
	v_dual_mul_f32 v81, v77, v87 :: v_dual_mul_f32 v82, v78, v87
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_dual_fmac_f32 v156, v81, v57 :: v_dual_mul_f32 v57, v75, v87
	v_mul_f32_e32 v81, v76, v87
	v_dual_fmac_f32 v157, v82, v58 :: v_dual_mul_f32 v58, v71, v87
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v154, v57, v59
	v_fmac_f32_e32 v155, v81, v60
	v_mul_f32_e32 v57, v72, v87
	v_cvt_f32_i32_e32 v59, v62
	v_fmac_f32_e32 v152, v58, v61
	v_mul_f32_e32 v58, v69, v87
	v_mul_f32_e32 v60, v70, v87
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_fmac_f32_e32 v153, v57, v59
	v_mul_f32_e32 v57, v77, v88
	v_cvt_f32_i32_e32 v50, v50
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v150, v58, v61 :: v_dual_fmac_f32 v151, v60, v62
	v_dual_mul_f32 v58, v78, v88 :: v_dual_fmac_f32 v145, v57, v49
	v_mul_f32_e32 v49, v75, v88
	v_mul_f32_e32 v57, v76, v88
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v52, v52
	v_fmac_f32_e32 v146, v58, v50
	v_mul_f32_e32 v50, v71, v88
	v_cvt_f32_i32_e32 v53, v53
	v_cvt_f32_i32_e32 v41, v41
	v_fmac_f32_e32 v144, v57, v52
	v_mul_f32_e32 v52, v70, v88
	v_cvt_f32_i32_e32 v42, v42
	v_dual_fmac_f32 v141, v50, v53 :: v_dual_mul_f32 v50, v69, v88
	v_fmac_f32_e32 v143, v49, v51
	v_mul_f32_e32 v49, v72, v88
	v_cvt_f32_i32_e32 v51, v54
	v_cvt_f32_i32_e32 v53, v55
	v_cvt_f32_i32_e32 v54, v56
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_fmac_f32_e32 v139, v50, v53
	s_wait_dscnt 0x2
	v_mul_f32_e32 v50, v78, v85
	v_fmac_f32_e32 v142, v49, v51
	v_dual_mul_f32 v49, v77, v85 :: v_dual_fmac_f32 v140, v52, v54
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v137, v50, v42 :: v_dual_mul_f32 v42, v71, v85
	v_fmac_f32_e32 v136, v49, v41
	v_mul_f32_e32 v41, v75, v85
	v_mul_f32_e32 v49, v76, v85
	v_cvt_f32_i32_e32 v34, v34
	v_fmac_f32_e32 v132, v42, v45
	v_mul_f32_e32 v42, v69, v85
	v_fmac_f32_e32 v134, v41, v43
	v_fmac_f32_e32 v135, v49, v44
	v_mul_f32_e32 v41, v72, v85
	v_cvt_f32_i32_e32 v43, v46
	v_cvt_f32_i32_e32 v45, v47
	v_mul_f32_e32 v44, v70, v85
	v_cvt_f32_i32_e32 v46, v48
	v_cvt_f32_i32_e32 v35, v35
	v_fmac_f32_e32 v133, v41, v43
	v_dual_mul_f32 v41, v77, v86 :: v_dual_fmac_f32 v130, v42, v45
	v_mul_f32_e32 v42, v78, v86
	v_fmac_f32_e32 v131, v44, v46
	v_cvt_f32_i32_e32 v36, v36
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v128, v41, v33 :: v_dual_mul_f32 v33, v75, v86
	v_mul_f32_e32 v41, v76, v86
	v_fmac_f32_e32 v129, v42, v34
	v_mul_f32_e32 v34, v71, v86
	v_cvt_f32_i32_e32 v37, v37
	v_fmac_f32_e32 v126, v33, v35
	v_fmac_f32_e32 v127, v41, v36
	v_mul_f32_e32 v33, v72, v86
	v_cvt_f32_i32_e32 v35, v38
	v_fmac_f32_e32 v124, v34, v37
	v_mul_f32_e32 v34, v69, v86
	v_mul_f32_e32 v36, v70, v86
	v_cvt_f32_i32_e32 v37, v39
	v_cvt_f32_i32_e32 v38, v40
	v_fmac_f32_e32 v125, v33, v35
	s_wait_dscnt 0x1
	v_mul_f32_e32 v33, v77, v79
	v_cvt_f32_i32_e32 v25, v25
	v_dual_fmac_f32 v122, v34, v37 :: v_dual_fmac_f32 v123, v36, v38
	v_mul_f32_e32 v34, v78, v79
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v120, v33, v25 :: v_dual_mul_f32 v25, v75, v79
	v_mul_f32_e32 v33, v76, v79
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_dual_fmac_f32 v121, v34, v26 :: v_dual_mul_f32 v26, v71, v79
	v_cvt_f32_i32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v118, v25, v27
	v_fmac_f32_e32 v119, v33, v28
	v_mul_f32_e32 v25, v72, v79
	v_cvt_f32_i32_e32 v27, v30
	v_fmac_f32_e32 v116, v26, v29
	v_mul_f32_e32 v26, v69, v79
	v_mul_f32_e32 v28, v70, v79
	v_cvt_f32_i32_e32 v29, v31
	v_cvt_f32_i32_e32 v30, v32
	v_fmac_f32_e32 v117, v25, v27
	v_mul_f32_e32 v25, v77, v80
	v_cvt_f32_i32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v114, v26, v29 :: v_dual_fmac_f32 v115, v28, v30
	v_mul_f32_e32 v26, v78, v80
	v_cvt_f32_i32_e32 v18, v18
	v_dual_fmac_f32 v112, v25, v17 :: v_dual_mul_f32 v17, v75, v80
	v_mul_f32_e32 v25, v76, v80
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_dual_fmac_f32 v113, v26, v18 :: v_dual_mul_f32 v18, v71, v80
	v_cvt_f32_i32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v110, v17, v19
	v_fmac_f32_e32 v111, v25, v20
	v_mul_f32_e32 v17, v72, v80
	v_cvt_f32_i32_e32 v19, v22
	v_fmac_f32_e32 v108, v18, v21
	v_mul_f32_e32 v18, v69, v80
	v_cvt_f32_i32_e32 v21, v23
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mul_f32 v20, v70, v80 :: v_dual_fmac_f32 v109, v17, v19
	v_cvt_f32_i32_e32 v22, v24
	s_wait_dscnt 0x0
	v_mul_f32_e32 v17, v77, v73
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v106, v18, v21
	v_mul_f32_e32 v18, v78, v73
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v104, v17, v9
	v_mul_f32_e32 v9, v75, v73
	v_mul_f32_e32 v17, v76, v73
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v105, v18, v10 :: v_dual_mul_f32 v10, v71, v73
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v103, v17, v12 :: v_dual_mul_f32 v12, v70, v73
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v100, v10, v13
	v_mul_f32_e32 v10, v69, v73
	v_dual_fmac_f32 v102, v9, v11 :: v_dual_mul_f32 v9, v72, v73
	v_cvt_f32_i32_e32 v11, v14
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v77, v74 :: v_dual_mul_f32 v16, v78, v74
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v101, v9, v11
	s_barrier_signal -1
	v_dual_fmac_f32 v96, v15, v1 :: v_dual_fmac_f32 v97, v16, v2
	v_mul_f32_e32 v1, v75, v74
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v98, v10, v13 :: v_dual_fmac_f32 v99, v12, v14
	v_dual_mul_f32 v3, v76, v74 :: v_dual_mul_f32 v10, v70, v74
	v_cvt_f32_i32_e32 v4, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v94, v1, v2
	v_mul_f32_e32 v2, v72, v74
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_mul_f32_e32 v9, v69, v74
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v1, v71, v74
	v_fmac_f32_e32 v107, v20, v22
	v_dual_fmac_f32 v95, v3, v4 :: v_dual_fmac_f32 v92, v2, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v91, v9, v7 :: v_dual_fmac_f32 v90, v10, v8
	v_fmac_f32_e32 v93, v1, v5
	s_xor_b32 s6, s27, -1
	s_mov_b32 s29, 1
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s6
	s_mov_b32 s28, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_424
.LBB1_427:                              ;   Parent Loop BB1_425 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s6, s29, s24
	s_mov_b32 s31, s7
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s30, s6, 1
	v_add_nc_u32_e32 v77, s26, v158
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[30:31], s[30:31], 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v160, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s31, v161, vcc_lo
	s_mul_u64 s[30:31], s[6:7], s[10:11]
	ds_load_2addr_stride64_b64 v[71:74], v77 offset0:6 offset1:7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	global_load_b128 v[1:4], v[69:70], off
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[14:15], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s29, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s31, 0, s29
	v_add_co_u32 v7, s29, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s31, 0, s29
	s_clause 0x1
	global_load_b64 v[81:82], v[5:6], off offset:40
	global_load_b64 v[83:84], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[5:8], v77 offset1:1
	ds_load_2addr_stride64_b64 v[9:12], v77 offset0:2 offset1:3
	ds_load_2addr_stride64_b64 v[13:16], v77 offset0:4 offset1:5
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v2, 0x88888888, v2
	v_xor_b32_e32 v1, 0x88888888, v1
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v76, 0, v2, s2
	v_cndmask_b32_e64 v75, 0, v1, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v80, 0, v4, s2
	v_cndmask_b32_e64 v79, 0, v3, s2
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[75:76], v[5:6], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[75:76], v[7:8], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[75:76], v[9:10], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[75:76], v[11:12], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[75:76], v[13:14], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[75:76], v[15:16], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[75:76], v[71:72], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[75:76], v[73:74], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v171, 0x100, v77
	ds_load_2addr_b64 v[71:74], v77 offset0:32 offset1:96
	ds_load_2addr_b64 v[75:78], v77 offset0:160 offset1:224
	ds_load_2addr_stride64_b64 v[85:88], v171 offset0:4 offset1:5
	ds_load_2addr_stride64_b64 v[171:174], v171 offset0:6 offset1:7
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[57:64], v[79:80], v[71:72], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[79:80], v[73:74], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[79:80], v[75:76], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[79:80], v[77:78], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[79:80], v[85:86], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[79:80], v[87:88], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[79:80], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[79:80], v[173:174], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v82, s0
	v_cndmask_b32_e64 v71, 0, v81, s0
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_b64 v148, v[71:72] offset:8192
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v84, s1
	v_cndmask_b32_e64 v71, 0, v83, s1
	ds_store_b64 v149, v[71:72] offset:8192
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_429
; %bb.428:                              ;   in Loop: Header=BB1_427 Depth=2
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[71:72], null, s6, s9, v[67:68]
	s_mul_u64 s[30:31], s[6:7], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[14:15], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[75:76], null, 0x48, v159, s[30:31]
	v_mad_co_u64_u32 v[72:73], null, s6, s12, v[72:73]
	v_add_co_u32 v73, s6, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, s31, 0, s6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[71:72], 6, v[71:72]
	v_add_co_u32 v71, vcc_lo, v162, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, v163, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v75, v168
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v76, vcc_lo
	s_clause 0x3
	global_load_u8 v77, v[71:72], off offset:16
	global_load_u8 v78, v[71:72], off offset:32
	global_load_u8 v79, v[71:72], off offset:48
	global_load_u8 v80, v[71:72], off
	v_add_co_u32 v71, s6, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, s31, 0, s6
	s_clause 0x2
	global_load_b64 v[81:82], v[73:74], off offset:8
	global_load_b64 v[83:84], v[71:72], off offset:8
	global_load_b32 v169, v[75:76], off
	s_wait_loadcnt 0x5
	v_lshlrev_b32_e32 v72, 16, v78
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v73, 24, v79
	s_wait_loadcnt 0x3
	v_lshl_or_b32 v71, v80, 8, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_or3_b32 v170, v71, v72, v73
.LBB1_429:                              ; %.preheader416.i
                                        ;   in Loop: Header=BB1_427 Depth=2
	global_load_b128 v[69:72], v[69:70], off offset:512
	v_add_nc_u32_e32 v177, 0, v158
	s_xor_b32 s6, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s13, 0x1400
	ds_load_2addr_stride64_b64 v[73:76], v177 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[77:80], v177 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[85:88], v177 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[171:174], v177 offset0:22 offset1:23
	s_cselect_b32 s29, s17, 0x1c00
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v70, 0x88888888, v70
	v_xor_b32_e32 v69, 0x88888888, v69
	v_xor_b32_e32 v71, 0x88888888, v71
	v_xor_b32_e32 v72, 0x88888888, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v70, 0, v70, s2
	v_cndmask_b32_e64 v69, 0, v69, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v175, 0, v71, s2
	v_cndmask_b32_e64 v176, 0, v72, s2
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[69:70], v[73:74], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[69:70], v[75:76], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[69:70], v[77:78], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[69:70], v[79:80], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[69:70], v[85:86], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[69:70], v[87:88], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[69:70], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[69:70], v[173:174], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v85, 0x100, v177
	ds_load_2addr_stride64_b64 v[69:72], v85 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[73:76], v85 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[77:80], v85 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[85:88], v85 offset0:22 offset1:23
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[57:64], v[175:176], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[175:176], v[71:72], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[175:176], v[73:74], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[175:176], v[75:76], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[175:176], v[77:78], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[175:176], v[79:80], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[175:176], v[85:86], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[175:176], v[87:88], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v69, s29, v166
	v_add_nc_u32_e32 v73, s26, v167
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_movk_i32 s26, 0x2000
	ds_load_2addr_b32 v[77:78], v69 offset1:2
	ds_load_2addr_b32 v[75:76], v69 offset0:4 offset1:6
	ds_load_2addr_b32 v[71:72], v69 offset0:8 offset1:10
	ds_load_2addr_b32 v[69:70], v69 offset0:12 offset1:14
	ds_load_2addr_b32 v[87:88], v73 offset1:32
	ds_load_2addr_b32 v[85:86], v73 offset0:64 offset1:96
	ds_load_2addr_b32 v[79:80], v73 offset0:128 offset1:160
	ds_load_2addr_b32 v[73:74], v73 offset0:192 offset1:224
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_426
; %bb.430:                              ; %.preheader417.i
                                        ;   in Loop: Header=BB1_427 Depth=2
	v_lshrrev_b32_e32 v171, v165, v170
	s_and_b32 s6, s28, exec_lo
	s_cselect_b32 s6, s13, 0x1400
	v_cndmask_b32_e64 v82, 0, v82, s0
	v_cndmask_b32_e64 v81, 0, v81, s0
	v_cvt_f32_f16_e64 v171, v171.l
	v_cndmask_b32_e64 v84, 0, v84, s1
	v_cndmask_b32_e64 v83, 0, v83, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v173, s6, v164
	s_cselect_b32 s6, s17, 0x1c00
	v_cndmask_b32_e64 v172, 0, v169, s3
	v_cndmask_b32_e64 v171, 0, v171, s4
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v174, s6, v164
	s_mov_b32 s26, 0
	ds_store_b64 v148, v[81:82]
	ds_store_b64 v149, v[83:84]
	ds_store_b32 v173, v172
	ds_store_b32 v174, v171
	s_branch .LBB1_426
.LBB1_431:                              ; %._crit_edge474.i
	v_mul_u32_u24_e32 v1, 0x500, v89
	v_lshlrev_b32_e32 v2, 2, v147
	v_and_b32_e32 v3, 0x7f, v0
	v_lshrrev_b32_e32 v5, 7, v0
	v_bfe_u32 v0, v0, 4, 3
	s_ashr_i32 s17, s16, 31
	v_add3_u32 v13, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[22:23], s[16:17]
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x500, v0, 0
	v_or_b32_e32 v0, s20, v3
	v_mad_i32_i24 v1, 0x50, v138, v13
	s_ashr_i32 s21, s20, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[20:21], 2
	ds_store_2addr_b32 v1, v156, v157 offset1:20
	ds_store_2addr_b32 v1, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v152, v153 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v150, v151 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v147
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[18:19], s[0:1]
	s_add_co_i32 s6, s20, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mul_lo_u32 v6, s8, v5
	v_add3_u32 v2, v2, v1, v4
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s6, s8
	s_mov_b32 s7, 0x31004000
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s16, 0x80
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s10
	s_mov_b32 s6, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v2
	s_cselect_b32 s1, -1, 0
	s_mov_b32 s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, s1
	v_cmp_gt_i32_e64 s0, s8, v0
	v_add_co_u32 v0, vcc_lo, s18, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s19, v8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_435
; %bb.432:
	v_or_b32_e32 v7, s16, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_434
; %bb.433:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s23, v7
	v_mad_co_u64_u32 v[7:8], null, s22, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s22, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v1, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v4, off
.LBB1_434:                              ; %Flow2118
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_435:                              ; %Flow2119
	v_add_lshl_u32 v3, v6, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_437
; %bb.436:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v3, s[4:7], null offen
.LBB1_437:
	ds_load_b32 v7, v2 offset:8
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, 0, 1, s1
	v_or_b32_e32 v6, 2, v5
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_441
; %bb.438:
	v_or_b32_e32 v8, s16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v8
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_440
; %bb.439:
	v_ashrrev_i32_e32 v10, 31, v8
	v_mul_lo_u32 v11, s23, v8
	v_mad_co_u64_u32 v[8:9], null, s22, v8, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v10, s22, v10
	v_add3_u32 v9, v9, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v0, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v1, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB1_440:                              ; %Flow2116
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_441:                              ; %Flow2117
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_443
; %bb.442:
	s_lshl_b32 s1, s8, 3
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s1 offen
.LBB1_443:
	ds_load_b32 v8, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v7, 4, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_447
; %bb.444:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v9, s16, v7
	v_cmp_gt_i32_e32 vcc_lo, s10, v9
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_446
; %bb.445:
	v_ashrrev_i32_e32 v11, 31, v9
	v_mul_lo_u32 v12, s23, v9
	v_mad_co_u64_u32 v[9:10], null, s22, v9, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, s22, v11
	v_add3_u32 v10, v10, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, v0, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB1_446:                              ; %Flow2114
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_447:                              ; %Flow2115
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_449
; %bb.448:
	s_lshl_b32 s1, s8, 4
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v3, s[4:7], s1 offen
.LBB1_449:
	ds_load_b32 v9, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v8, 6, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_453
; %bb.450:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v10, s16, v8
	v_cmp_gt_i32_e32 vcc_lo, s10, v10
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_452
; %bb.451:
	v_ashrrev_i32_e32 v12, 31, v10
	v_mul_lo_u32 v14, s23, v10
	v_mad_co_u64_u32 v[10:11], null, s22, v10, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s22, v12
	v_add3_u32 v11, v11, v12, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, vcc_lo, v0, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v1, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off
.LBB1_452:                              ; %Flow2112
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_453:                              ; %Flow2113
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_455
; %bb.454:
	s_mul_i32 s1, s8, 24
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s1 offen
.LBB1_455:
	ds_load_b32 v10, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 8, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_459
; %bb.456:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v11, s16, v9
	v_cmp_gt_i32_e32 vcc_lo, s10, v11
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_458
; %bb.457:
	v_ashrrev_i32_e32 v14, 31, v11
	v_mul_lo_u32 v15, s23, v11
	v_mad_co_u64_u32 v[11:12], null, s22, v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v14, s22, v14
	v_add3_u32 v12, v12, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v1, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v10, off
.LBB1_458:                              ; %Flow2110
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_459:                              ; %Flow2111
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_461
; %bb.460:
	s_lshl_b32 s1, s8, 5
	s_wait_dscnt 0x0
	buffer_store_b32 v10, v3, s[4:7], s1 offen
.LBB1_461:
	ds_load_b32 v11, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v10, 10, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_465
; %bb.462:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, s16, v10
	v_cmp_gt_i32_e32 vcc_lo, s10, v12
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_464
; %bb.463:
	v_ashrrev_i32_e32 v16, 31, v12
	v_mul_lo_u32 v17, s23, v12
	v_mad_co_u64_u32 v[14:15], null, s22, v12, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s22, v16
	v_add3_u32 v15, v15, v12, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v0, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v1, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v11, off
.LBB1_464:                              ; %Flow2108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_465:                              ; %Flow2109
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_467
; %bb.466:
	s_mul_i32 s1, s8, 40
	s_wait_dscnt 0x0
	buffer_store_b32 v11, v3, s[4:7], s1 offen
.LBB1_467:
	ds_load_b32 v12, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 12, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_471
; %bb.468:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v14, s16, v11
	v_cmp_gt_i32_e32 vcc_lo, s10, v14
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_470
; %bb.469:
	v_ashrrev_i32_e32 v16, 31, v14
	v_mul_lo_u32 v17, s23, v14
	v_mad_co_u64_u32 v[14:15], null, s22, v14, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v16, s22, v16
	v_add3_u32 v15, v15, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v0, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v1, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v12, off
.LBB1_470:                              ; %Flow2106
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_471:                              ; %Flow2107
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_473
; %bb.472:
	s_mul_i32 s1, s8, 48
	s_wait_dscnt 0x0
	buffer_store_b32 v12, v3, s[4:7], s1 offen
.LBB1_473:
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v12, 14, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB1_477
; %bb.474:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v15, s16, v12
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_476
; %bb.475:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_476:                              ; %Flow2104
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB1_477:                              ; %Flow2105
	v_mul_i32_i24_e32 v15, 0x50, v138
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_479
; %bb.478:
	s_mul_i32 s1, s8, 56
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_479:                              ; %.preheader.1.i42
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v13, v13, v15
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 16
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v145, v146 offset1:20
	ds_store_2addr_b32 v13, v143, v144 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v141, v142 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v139, v140 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_483
; %bb.480:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_482
; %bb.481:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_482:                              ; %Flow2102
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_483:                              ; %Flow2103
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_485
; %bb.484:
	s_lshl_b32 s2, s8, 6
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_485:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_489
; %bb.486:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_488
; %bb.487:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_488:                              ; %Flow2100
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_489:                              ; %Flow2101
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_491
; %bb.490:
	s_mul_i32 s2, s8, 0x48
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_491:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_495
; %bb.492:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_494
; %bb.493:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_494:                              ; %Flow2098
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_495:                              ; %Flow2099
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_497
; %bb.496:
	s_mul_i32 s2, s8, 0x50
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_497:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_501
; %bb.498:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_500
; %bb.499:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_500:                              ; %Flow2096
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_501:                              ; %Flow2097
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_503
; %bb.502:
	s_mul_i32 s2, s8, 0x58
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_503:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_507
; %bb.504:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_506
; %bb.505:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_506:                              ; %Flow2094
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_507:                              ; %Flow2095
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_509
; %bb.508:
	s_mul_i32 s2, s8, 0x60
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_509:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_513
; %bb.510:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_512
; %bb.511:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_512:                              ; %Flow2092
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_513:                              ; %Flow2093
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_515
; %bb.514:
	s_mul_i32 s2, s8, 0x68
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_515:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_519
; %bb.516:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_518
; %bb.517:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_518:                              ; %Flow2090
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_519:                              ; %Flow2091
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_521
; %bb.520:
	s_mul_i32 s2, s8, 0x70
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_521:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_525
; %bb.522:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_524
; %bb.523:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_524:                              ; %Flow2088
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_525:                              ; %Flow2089
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_527
; %bb.526:
	s_mul_i32 s1, s8, 0x78
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_527:                              ; %.preheader.2.i43
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 32
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v136, v137 offset1:20
	ds_store_2addr_b32 v13, v134, v135 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v132, v133 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v130, v131 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_531
; %bb.528:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_530
; %bb.529:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_530:                              ; %Flow2086
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_531:                              ; %Flow2087
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_533
; %bb.532:
	s_lshl_b32 s2, s8, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_533:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_537
; %bb.534:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_536
; %bb.535:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_536:                              ; %Flow2084
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_537:                              ; %Flow2085
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_539
; %bb.538:
	s_mul_i32 s2, s8, 0x88
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_539:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_543
; %bb.540:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_542
; %bb.541:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_542:                              ; %Flow2082
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_543:                              ; %Flow2083
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_545
; %bb.544:
	s_mul_i32 s2, s8, 0x90
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_545:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_549
; %bb.546:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_548
; %bb.547:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_548:                              ; %Flow2080
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_549:                              ; %Flow2081
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_551
; %bb.550:
	s_mul_i32 s2, s8, 0x98
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_551:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_555
; %bb.552:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_554
; %bb.553:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_554:                              ; %Flow2078
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_555:                              ; %Flow2079
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_557
; %bb.556:
	s_mul_i32 s2, s8, 0xa0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_557:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_561
; %bb.558:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_560
; %bb.559:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_560:                              ; %Flow2076
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_561:                              ; %Flow2077
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_563
; %bb.562:
	s_mul_i32 s2, s8, 0xa8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_563:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_567
; %bb.564:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_566
; %bb.565:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_566:                              ; %Flow2074
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_567:                              ; %Flow2075
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_569
; %bb.568:
	s_mul_i32 s2, s8, 0xb0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_569:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_573
; %bb.570:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_572
; %bb.571:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_572:                              ; %Flow2072
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_573:                              ; %Flow2073
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_575
; %bb.574:
	s_mul_i32 s1, s8, 0xb8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_575:                              ; %.preheader.3.i44
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 48
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v128, v129 offset1:20
	ds_store_2addr_b32 v13, v126, v127 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v124, v125 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v122, v123 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_579
; %bb.576:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_578
; %bb.577:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_578:                              ; %Flow2070
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_579:                              ; %Flow2071
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_581
; %bb.580:
	s_mul_i32 s2, s8, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_581:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_585
; %bb.582:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_584
; %bb.583:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_584:                              ; %Flow2068
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_585:                              ; %Flow2069
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_587
; %bb.586:
	s_mul_i32 s2, s8, 0xc8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_587:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_591
; %bb.588:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_590
; %bb.589:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_590:                              ; %Flow2066
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_591:                              ; %Flow2067
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_593
; %bb.592:
	s_mul_i32 s2, s8, 0xd0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_593:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_597
; %bb.594:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_596
; %bb.595:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_596:                              ; %Flow2064
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_597:                              ; %Flow2065
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_599
; %bb.598:
	s_mul_i32 s2, s8, 0xd8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_599:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_603
; %bb.600:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_602
; %bb.601:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_602:                              ; %Flow2062
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_603:                              ; %Flow2063
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_605
; %bb.604:
	s_mul_i32 s2, s8, 0xe0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_605:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_609
; %bb.606:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_608
; %bb.607:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_608:                              ; %Flow2060
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_609:                              ; %Flow2061
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_611
; %bb.610:
	s_mul_i32 s2, s8, 0xe8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_611:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_615
; %bb.612:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_614
; %bb.613:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_614:                              ; %Flow2058
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_615:                              ; %Flow2059
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_617
; %bb.616:
	s_mul_i32 s2, s8, 0xf0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_617:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_621
; %bb.618:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_620
; %bb.619:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_620:                              ; %Flow2056
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_621:                              ; %Flow2057
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_623
; %bb.622:
	s_mul_i32 s1, s8, 0xf8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_623:                              ; %.preheader414.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 64
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v120, v121 offset1:20
	ds_store_2addr_b32 v13, v118, v119 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v116, v117 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v114, v115 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_627
; %bb.624:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_626
; %bb.625:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_626:                              ; %Flow2054
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_627:                              ; %Flow2055
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_629
; %bb.628:
	s_lshl_b32 s2, s8, 8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_629:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_633
; %bb.630:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_632
; %bb.631:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_632:                              ; %Flow2052
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_633:                              ; %Flow2053
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_635
; %bb.634:
	s_mul_i32 s2, s8, 0x108
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_635:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_639
; %bb.636:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_638
; %bb.637:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_638:                              ; %Flow2050
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_639:                              ; %Flow2051
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_641
; %bb.640:
	s_mul_i32 s2, s8, 0x110
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_641:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_645
; %bb.642:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_644
; %bb.643:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_644:                              ; %Flow2048
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_645:                              ; %Flow2049
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_647
; %bb.646:
	s_mul_i32 s2, s8, 0x118
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_647:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_651
; %bb.648:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_650
; %bb.649:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_650:                              ; %Flow2046
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_651:                              ; %Flow2047
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_653
; %bb.652:
	s_mul_i32 s2, s8, 0x120
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_653:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_657
; %bb.654:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_656
; %bb.655:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_656:                              ; %Flow2044
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_657:                              ; %Flow2045
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_659
; %bb.658:
	s_mul_i32 s2, s8, 0x128
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_659:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_663
; %bb.660:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_662
; %bb.661:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_662:                              ; %Flow2042
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_663:                              ; %Flow2043
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_665
; %bb.664:
	s_mul_i32 s2, s8, 0x130
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_665:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_669
; %bb.666:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_668
; %bb.667:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_668:                              ; %Flow2040
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_669:                              ; %Flow2041
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_671
; %bb.670:
	s_mul_i32 s1, s8, 0x138
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_671:                              ; %.preheader.1.1.i45
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 0x50
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v112, v113 offset1:20
	ds_store_2addr_b32 v13, v110, v111 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v108, v109 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v106, v107 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_675
; %bb.672:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_674
; %bb.673:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_674:                              ; %Flow2038
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_675:                              ; %Flow2039
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_677
; %bb.676:
	s_mul_i32 s2, s8, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_677:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_681
; %bb.678:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_680
; %bb.679:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_680:                              ; %Flow2036
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_681:                              ; %Flow2037
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_683
; %bb.682:
	s_mul_i32 s2, s8, 0x148
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_683:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_687
; %bb.684:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_686
; %bb.685:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_686:                              ; %Flow2034
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_687:                              ; %Flow2035
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_689
; %bb.688:
	s_mul_i32 s2, s8, 0x150
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_689:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_693
; %bb.690:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_692
; %bb.691:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_692:                              ; %Flow2032
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_693:                              ; %Flow2033
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_695
; %bb.694:
	s_mul_i32 s2, s8, 0x158
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_695:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_699
; %bb.696:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_698
; %bb.697:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_698:                              ; %Flow2030
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_699:                              ; %Flow2031
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_701
; %bb.700:
	s_mul_i32 s2, s8, 0x160
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_701:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_705
; %bb.702:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_704
; %bb.703:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_704:                              ; %Flow2028
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_705:                              ; %Flow2029
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_707
; %bb.706:
	s_mul_i32 s2, s8, 0x168
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_707:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_711
; %bb.708:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_710
; %bb.709:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_710:                              ; %Flow2026
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_711:                              ; %Flow2027
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_713
; %bb.712:
	s_mul_i32 s2, s8, 0x170
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_713:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_717
; %bb.714:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_716
; %bb.715:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_716:                              ; %Flow2024
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_717:                              ; %Flow2025
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_719
; %bb.718:
	s_mul_i32 s1, s8, 0x178
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_719:                              ; %.preheader.2.1.i46
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 0x60
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v104, v105 offset1:20
	ds_store_2addr_b32 v13, v102, v103 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v100, v101 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v98, v99 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_723
; %bb.720:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_722
; %bb.721:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_722:                              ; %Flow2022
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_723:                              ; %Flow2023
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_725
; %bb.724:
	s_mul_i32 s2, s8, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_725:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_729
; %bb.726:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_728
; %bb.727:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_728:                              ; %Flow2020
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_729:                              ; %Flow2021
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_731
; %bb.730:
	s_mul_i32 s2, s8, 0x188
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_731:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_735
; %bb.732:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_734
; %bb.733:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_734:                              ; %Flow2018
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_735:                              ; %Flow2019
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_737
; %bb.736:
	s_mul_i32 s2, s8, 0x190
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_737:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_741
; %bb.738:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_740
; %bb.739:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_740:                              ; %Flow2016
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_741:                              ; %Flow2017
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_743
; %bb.742:
	s_mul_i32 s2, s8, 0x198
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_743:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_747
; %bb.744:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_746
; %bb.745:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_746:                              ; %Flow2014
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_747:                              ; %Flow2015
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_749
; %bb.748:
	s_mul_i32 s2, s8, 0x1a0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_749:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_753
; %bb.750:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_752
; %bb.751:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_752:                              ; %Flow2012
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_753:                              ; %Flow2013
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_755
; %bb.754:
	s_mul_i32 s2, s8, 0x1a8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_755:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_759
; %bb.756:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_758
; %bb.757:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_758:                              ; %Flow2010
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_759:                              ; %Flow2011
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_761
; %bb.760:
	s_mul_i32 s2, s8, 0x1b0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB1_761:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_765
; %bb.762:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB1_764
; %bb.763:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s23, v15
	v_mad_co_u64_u32 v[15:16], null, s22, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s22, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB1_764:                              ; %Flow2008
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB1_765:                              ; %Flow2009
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_767
; %bb.766:
	s_mul_i32 s1, s8, 0x1b8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB1_767:                              ; %.preheader.3.1.i47
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s16, 0x70
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v96, v97 offset1:20
	ds_store_2addr_b32 v13, v94, v95 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v93, v92 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v91, v90 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_771
; %bb.768:
	v_or_b32_e32 v5, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v5
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_770
; %bb.769:
	v_ashrrev_i32_e32 v16, 31, v5
	v_mul_lo_u32 v17, s23, v5
	v_mad_co_u64_u32 v[14:15], null, s22, v5, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v5, s22, v16
	v_add3_u32 v15, v15, v5, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v0, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v1, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v13, off
.LBB1_770:                              ; %Flow2006
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_771:                              ; %Flow2007
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_773
; %bb.772:
	s_mul_i32 s2, s8, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v13, v3, s[4:7], s2 offen
.LBB1_773:
	ds_load_b32 v5, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_777
; %bb.774:
	v_or_b32_e32 v6, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_776
; %bb.775:
	v_ashrrev_i32_e32 v15, 31, v6
	v_mul_lo_u32 v16, s23, v6
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[13:14], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s22, v15
	v_add3_u32 v14, v14, v6, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v1, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v5, off
.LBB1_776:                              ; %Flow2004
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_777:                              ; %Flow2005
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_779
; %bb.778:
	s_mul_i32 s2, s8, 0x1c8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_779:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_783
; %bb.780:
	v_or_b32_e32 v6, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_782
; %bb.781:
	v_ashrrev_i32_e32 v13, 31, v6
	v_mul_lo_u32 v14, s23, v6
	v_mad_co_u64_u32 v[6:7], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v13, s22, v13
	v_add3_u32 v7, v7, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB1_782:                              ; %Flow2002
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_783:                              ; %Flow2003
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_785
; %bb.784:
	s_mul_i32 s2, s8, 0x1d0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_785:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_789
; %bb.786:
	v_or_b32_e32 v6, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_788
; %bb.787:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v13, s23, v6
	v_mad_co_u64_u32 v[6:7], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s22, v8
	v_add3_u32 v7, v7, v8, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB1_788:                              ; %Flow2000
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_789:                              ; %Flow2001
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_791
; %bb.790:
	s_mul_i32 s2, s8, 0x1d8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_791:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_795
; %bb.792:
	v_or_b32_e32 v6, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_794
; %bb.793:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s23, v6
	v_mad_co_u64_u32 v[6:7], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s22, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB1_794:                              ; %Flow1998
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_795:                              ; %Flow1999
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_797
; %bb.796:
	s_mul_i32 s2, s8, 0x1e0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_797:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_801
; %bb.798:
	v_add_nc_u32_e32 v6, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_800
; %bb.799:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s23, v6
	v_mad_co_u64_u32 v[6:7], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s22, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB1_800:                              ; %Flow1996
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_801:                              ; %Flow1997
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_803
; %bb.802:
	s_mul_i32 s2, s8, 0x1e8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_803:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_807
; %bb.804:
	v_add_nc_u32_e32 v6, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB1_806
; %bb.805:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s23, v6
	v_mad_co_u64_u32 v[6:7], null, s22, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s22, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB1_806:                              ; %Flow1994
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB1_807:                              ; %Flow1995
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_809
; %bb.808:
	s_mul_i32 s2, s8, 0x1f0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB1_809:
	ds_load_b32 v2, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB1_813
; %bb.810:
	v_add_nc_u32_e32 v4, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v4
	s_and_b32 s1, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_812
; %bb.811:
	v_ashrrev_i32_e32 v6, 31, v4
	v_mul_lo_u32 v7, s23, v4
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[4:5], null, s22, v4, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s22, v6
	v_add3_u32 v5, v5, v6, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v1, v5, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v2, off
.LBB1_812:                              ; %Flow1991
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB1_813:                              ; %Flow1993
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_815
; %bb.814:
	s_mul_i32 s0, s8, 0x1f8
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v3, s[4:7], s0 offen
.LBB1_815:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, -1
	s_barrier_wait -1
.LBB1_816:                              ; %Flow2131
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_818
; %bb.817:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_818:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
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
		.amdhsa_next_free_vgpr 179
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_vgpr, 179
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.numbered_sgpr, 38
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 36804
; TotalNumSgprs: 40
; NumVgprs: 179
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 179
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
	s_load_b96 s[8:10], s[0:1], 0x18
	s_lshl_b32 s18, ttmp9, 7
	s_lshl_b32 s14, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s18, s8
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s14, s10
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_407
; %bb.1:                                ; %.preheader427.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	v_lshrrev_b32_e32 v5, 1, v0
	v_and_b32_e32 v4, 1, v0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_9
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s14, v5
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v3, 2, v4
	v_mov_b32_e32 v7, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB2_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[6:7], null, 0x48, v1, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, v6, v3
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	global_load_b32 v7, v[6:7], off
.LBB2_4:                                ; %.preheader424.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, s18, v5
	s_add_co_i32 s1, s8, -1
	s_ashr_i32 s3, s8, 31
	s_mov_b32 s2, s8
	s_ashr_i32 s13, s9, 31
	v_min_i32_e32 v8, s1, v6
	s_mov_b32 s12, s9
	v_lshlrev_b32_e32 v9, 3, v5
	s_mul_u64 s[2:3], s[12:13], s[2:3]
	s_mov_b32 s1, exec_lo
	v_ashrrev_i32_e32 v1, 4, v8
	s_lshr_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v10, 15, v8
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	v_add3_u32 v8, 0, v9, v3
	v_lshlrev_b64_e32 v[1:2], 6, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b32 v8, v7 offset:4096
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s3, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v11, vcc_lo
                                        ; implicit-def: $vgpr1_lo16
	v_cmpx_ne_u32_e32 0, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off offset:48
	global_load_d16_hi_u8 v1, v[2:3], off offset:32
                                        ; implicit-def: $vgpr2_vgpr3
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB2_6:                                ; %Flow994
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB2_8
; %bb.7:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off
	global_load_d16_hi_u8 v1, v[2:3], off offset:16
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB2_8:                                ; %._crit_edge.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cmp_gt_i32_e32 vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	ds_store_b32 v8, v1 offset:6144
.LBB2_9:                                ; %Flow995
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v1, 3, v0
	s_add_co_i32 s3, s10, -1
	v_lshlrev_b32_e32 v10, 4, v0
	v_lshrrev_b32_e32 v145, 5, v0
	v_add_nc_u32_e32 v8, s14, v3
	v_lshlrev_b32_e32 v1, 3, v1
	v_bfe_u32 v11, v0, 1, 1
	v_and_b32_e32 v10, 16, v10
	s_mov_b32 s2, -1
	v_add_nc_u32_e32 v9, 64, v8
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v2, s3, v8
	v_and_or_b32 v12, v145, 6, v11
	v_and_or_b32 v3, v3, 15, v10
	v_or_b32_e32 v10, 8, v145
	v_min_i32_e32 v6, s3, v9
	v_mad_co_u64_u32 v[65:66], null, 0x48, v2, v[1:2]
	v_cmp_gt_i32_e64 s0, s10, v8
	v_lshlrev_b32_e32 v3, 3, v3
	v_and_or_b32 v10, v10, 14, v11
	v_mad_co_u64_u32 v[66:67], null, 0x48, v6, v[1:2]
	v_cmp_gt_i32_e64 s1, s10, v9
	s_cmp_gt_i32 s9, 0xff
	v_lshl_or_b32 v11, v12, 8, v3
	v_lshl_or_b32 v3, v10, 8, v3
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b64 v[1:2], v65, s[6:7] offset:8
	global_load_b64 v[6:7], v66, s[6:7] offset:8
	v_add_nc_u32_e32 v148, 0, v11
	v_add_nc_u32_e32 v149, 0, v3
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v7, 0, v7, s1
	v_cndmask_b32_e64 v6, 0, v6, s1
	ds_store_b64 v148, v[1:2]
	ds_store_b64 v149, v[6:7]
	v_bfe_u32 v1, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v147, 3, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_11
; %bb.10:                               ; %._crit_edge.._crit_edge479_crit_edge.i
	v_lshlrev_b32_e32 v13, 3, v1
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	s_mov_b32 s2, 0
	s_branch .LBB2_12
.LBB2_11:
                                        ; implicit-def: $sgpr12_sgpr13
                                        ; implicit-def: $vgpr13
.LBB2_12:                               ; %Flow991
	v_dual_mov_b32 v89, 0 :: v_dual_and_b32 v146, 15, v0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v157, 0
	v_mov_b32_e32 v156, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_22
; %bb.13:                               ; %.preheader423.lr.ph.i
	v_dual_mov_b32 v68, 0 :: v_dual_lshlrev_b32 v3, 4, v145
	s_add_co_i32 s12, s8, -16
	s_ashr_i32 s22, s9, 6
	v_and_b32_e32 v6, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v170, v68 :: v_dual_add_nc_u32 v1, s18, v3
	s_ashr_i32 s23, s22, 31
	v_dual_mov_b32 v157, v68 :: v_dual_add_nc_u32 v8, s18, v5
	v_cmp_gt_i32_e64 s2, s8, v1
	s_lshl_b32 s24, s22, 9
	s_lshr_b64 s[22:23], s[22:23], 23
	s_add_co_i32 s19, s8, -1
	v_mov_b32_e32 v169, v68
	v_cndmask_b32_e64 v1, s12, v1, s2
	v_dual_mov_b32 v155, v68 :: v_dual_lshlrev_b32 v158, 3, v6
	v_dual_mov_b32 v153, v68 :: v_dual_lshlrev_b32 v6, 4, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_ashrrev_i32_e32 v1, 4, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v11, s19, v8
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	s_mov_b32 s21, 0
	v_ashrrev_i32_e32 v9, 31, v1
	v_mul_lo_u32 v10, s22, v1
	v_mad_co_u64_u32 v[1:2], null, s24, v1, s[4:5]
	s_mov_b32 s20, s9
	v_dual_mov_b32 v156, v68 :: v_dual_add_nc_u32 v7, s14, v5
	v_mul_lo_u32 v9, s24, v9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[22:23], s[20:21], s[12:13]
	v_mov_b32_e32 v154, v68
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[22:23], s[22:23], 1
	v_add_co_u32 v160, vcc_lo, v1, v6
	v_and_b32_e32 v1, 15, v11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	v_add3_u32 v2, v10, v2, v9
	v_min_i32_e32 v159, s3, v7
	s_ashr_i32 s15, s9, 31
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v162, s3, s22, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, 0, v2, vcc_lo
	v_dual_mov_b32 v151, v68 :: v_dual_lshlrev_b32 v2, 3, v5
	v_dual_mov_b32 v152, v68 :: v_dual_lshlrev_b32 v5, 2, v4
	v_or_b32_e32 v1, v147, v3
	s_lshr_b32 s15, s15, 24
	v_ashrrev_i32_e32 v67, 4, v11
	v_add_co_ci_u32_e64 v163, null, s23, 0, s3
	v_cmp_gt_i32_e64 s3, s10, v7
	v_add3_u32 v164, 0, v2, v5
	v_cmp_gt_i32_e64 s4, s8, v8
	v_dual_mov_b32 v150, v68 :: v_dual_lshlrev_b32 v165, 4, v4
	v_lshl_add_u32 v166, v1, 3, 0
	v_lshl_add_u32 v167, v146, 3, 0
	v_dual_mov_b32 v119, v68 :: v_dual_lshlrev_b32 v168, 2, v4
	v_dual_mov_b32 v120, v68 :: v_dual_mov_b32 v117, v68
	v_dual_mov_b32 v118, v68 :: v_dual_mov_b32 v115, v68
	v_dual_mov_b32 v116, v68 :: v_dual_mov_b32 v113, v68
	v_dual_mov_b32 v114, v68 :: v_dual_mov_b32 v143, v68
	v_dual_mov_b32 v144, v68 :: v_dual_mov_b32 v141, v68
	v_dual_mov_b32 v142, v68 :: v_dual_mov_b32 v139, v68
	v_dual_mov_b32 v140, v68 :: v_dual_mov_b32 v137, v68
	v_dual_mov_b32 v138, v68 :: v_dual_mov_b32 v111, v68
	v_dual_mov_b32 v112, v68 :: v_dual_mov_b32 v109, v68
	v_dual_mov_b32 v110, v68 :: v_dual_mov_b32 v107, v68
	v_dual_mov_b32 v108, v68 :: v_dual_mov_b32 v105, v68
	v_dual_mov_b32 v106, v68 :: v_dual_mov_b32 v135, v68
	v_dual_mov_b32 v136, v68 :: v_dual_mov_b32 v133, v68
	v_dual_mov_b32 v134, v68 :: v_dual_mov_b32 v131, v68
	v_dual_mov_b32 v132, v68 :: v_dual_mov_b32 v129, v68
	v_dual_mov_b32 v130, v68 :: v_dual_mov_b32 v103, v68
	v_dual_mov_b32 v104, v68 :: v_dual_mov_b32 v101, v68
	v_dual_mov_b32 v102, v68 :: v_dual_mov_b32 v99, v68
	v_dual_mov_b32 v100, v68 :: v_dual_mov_b32 v97, v68
	v_dual_mov_b32 v98, v68 :: v_dual_mov_b32 v127, v68
	v_dual_mov_b32 v128, v68 :: v_dual_mov_b32 v125, v68
	v_dual_mov_b32 v126, v68 :: v_dual_mov_b32 v123, v68
	v_dual_mov_b32 v124, v68 :: v_dual_mov_b32 v121, v68
	v_dual_mov_b32 v122, v68 :: v_dual_mov_b32 v95, v68
	v_dual_mov_b32 v96, v68 :: v_dual_mov_b32 v93, v68
	v_dual_mov_b32 v94, v68 :: v_dual_mov_b32 v91, v68
	v_dual_mov_b32 v92, v68 :: v_dual_mov_b32 v89, v68
	v_mov_b32_e32 v90, v68
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, s15
	s_ashr_i32 s15, s8, 4
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s9, 8
	s_ashr_i32 s5, s15, 31
	s_movk_i32 s19, 0x1000
	s_movk_i32 s22, 0x1800
	s_mov_b32 s26, 0
	s_mov_b32 s23, 0
	s_branch .LBB2_15
.LBB2_14:                               ;   in Loop: Header=BB2_15 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_21
.LBB2_15:                               ; %.preheader423.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_17 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s24, s23, 1
	s_add_co_i32 s23, s23, 1
	s_mov_b32 s28, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, s9
	s_mov_b32 s27, -1
	s_cselect_b32 s25, -1, 0
	s_mov_b32 s29, s21
	s_branch .LBB2_17
.LBB2_16:                               ;   in Loop: Header=BB2_17 Depth=2
	s_wait_dscnt 0x3
	v_dual_mul_f32 v79, v77, v87 :: v_dual_mul_f32 v80, v78, v87
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_dual_fmac_f32 v156, v79, v57 :: v_dual_mul_f32 v79, v76, v87
	v_dual_fmac_f32 v157, v80, v58 :: v_dual_mul_f32 v58, v71, v87
	v_mul_f32_e32 v57, v75, v87
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v155, v79, v60 :: v_dual_mul_f32 v60, v70, v87
	v_fmac_f32_e32 v152, v58, v61
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v57, v59
	v_dual_mul_f32 v57, v72, v87 :: v_dual_mul_f32 v58, v69, v87
	v_cvt_f32_i32_e32 v59, v62
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v153, v57, v59 :: v_dual_fmac_f32 v150, v58, v61
	v_mul_f32_e32 v57, v77, v88
	v_dual_fmac_f32 v151, v60, v62 :: v_dual_mul_f32 v58, v78, v88
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v53, v53
	v_fmac_f32_e32 v143, v57, v49
	v_mul_f32_e32 v49, v75, v88
	v_dual_mul_f32 v57, v76, v88 :: v_dual_fmac_f32 v144, v58, v50
	v_mul_f32_e32 v50, v71, v88
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v57, v52
	v_mul_f32_e32 v52, v70, v88
	v_dual_fmac_f32 v139, v50, v53 :: v_dual_mul_f32 v50, v69, v88
	v_fmac_f32_e32 v141, v49, v51
	v_mul_f32_e32 v49, v72, v88
	v_cvt_f32_i32_e32 v51, v54
	v_cvt_f32_i32_e32 v53, v55
	v_cvt_f32_i32_e32 v54, v56
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_fmac_f32_e32 v137, v50, v53
	s_wait_dscnt 0x2
	v_mul_f32_e32 v50, v78, v85
	v_fmac_f32_e32 v140, v49, v51
	v_dual_mul_f32 v49, v77, v85 :: v_dual_fmac_f32 v138, v52, v54
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v136, v50, v42
	v_mul_f32_e32 v42, v71, v85
	v_fmac_f32_e32 v135, v49, v41
	v_mul_f32_e32 v49, v76, v85
	v_mul_f32_e32 v41, v75, v85
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_fmac_f32_e32 v134, v49, v44
	v_dual_fmac_f32 v133, v41, v43 :: v_dual_mul_f32 v44, v70, v85
	v_mul_f32_e32 v41, v72, v85
	v_cvt_f32_i32_e32 v43, v46
	v_fmac_f32_e32 v131, v42, v45
	v_mul_f32_e32 v42, v69, v85
	v_cvt_f32_i32_e32 v45, v47
	v_cvt_f32_i32_e32 v46, v48
	v_fmac_f32_e32 v132, v41, v43
	v_mul_f32_e32 v41, v77, v86
	v_cvt_f32_i32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v129, v42, v45 :: v_dual_fmac_f32 v130, v44, v46
	v_dual_mul_f32 v42, v78, v86 :: v_dual_fmac_f32 v127, v41, v33
	v_mul_f32_e32 v41, v76, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v42, v34
	v_mul_f32_e32 v34, v71, v86
	v_dual_mul_f32 v33, v75, v86 :: v_dual_fmac_f32 v126, v41, v36
	v_mul_f32_e32 v36, v70, v86
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v33, v35
	v_mul_f32_e32 v33, v72, v86
	v_cvt_f32_i32_e32 v35, v38
	v_dual_fmac_f32 v123, v34, v37 :: v_dual_mul_f32 v34, v69, v86
	v_cvt_f32_i32_e32 v37, v39
	v_cvt_f32_i32_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v124, v33, v35
	s_wait_dscnt 0x1
	v_mul_f32_e32 v33, v77, v83
	v_cvt_f32_i32_e32 v29, v29
	v_dual_fmac_f32 v121, v34, v37 :: v_dual_fmac_f32 v122, v36, v38
	v_dual_mul_f32 v34, v78, v83 :: v_dual_fmac_f32 v119, v33, v25
	v_mul_f32_e32 v33, v76, v83
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v34, v26
	v_mul_f32_e32 v26, v71, v83
	v_dual_mul_f32 v25, v75, v83 :: v_dual_fmac_f32 v118, v33, v28
	v_mul_f32_e32 v28, v70, v83
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v25, v27
	v_mul_f32_e32 v25, v72, v83
	v_cvt_f32_i32_e32 v27, v30
	v_cvt_f32_i32_e32 v30, v32
	v_dual_fmac_f32 v115, v26, v29 :: v_dual_mul_f32 v26, v69, v83
	v_cvt_f32_i32_e32 v29, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v116, v25, v27
	v_mul_f32_e32 v25, v77, v84
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v113, v26, v29
	v_dual_mul_f32 v26, v78, v84 :: v_dual_fmac_f32 v111, v25, v17
	v_mul_f32_e32 v17, v75, v84
	v_mul_f32_e32 v25, v76, v84
	v_cvt_f32_i32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v26, v18
	v_mul_f32_e32 v18, v71, v84
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v110, v25, v20
	v_mul_f32_e32 v20, v70, v84
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v107, v18, v21 :: v_dual_mul_f32 v18, v69, v84
	v_fmac_f32_e32 v109, v17, v19
	v_mul_f32_e32 v17, v72, v84
	v_cvt_f32_i32_e32 v19, v22
	v_cvt_f32_i32_e32 v21, v23
	v_cvt_f32_i32_e32 v22, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v105, v18, v21
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v78, v73
	v_fmac_f32_e32 v108, v17, v19
	v_dual_mul_f32 v17, v77, v73 :: v_dual_fmac_f32 v106, v20, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v104, v18, v10
	v_mul_f32_e32 v10, v71, v73
	v_fmac_f32_e32 v103, v17, v9
	v_mul_f32_e32 v17, v76, v73
	v_mul_f32_e32 v9, v75, v73
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v102, v17, v12
	v_dual_fmac_f32 v101, v9, v11 :: v_dual_mul_f32 v12, v70, v73
	v_mul_f32_e32 v9, v72, v73
	v_cvt_f32_i32_e32 v11, v14
	v_fmac_f32_e32 v99, v10, v13
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v77, v74 :: v_dual_mul_f32 v16, v78, v74
	v_mul_f32_e32 v10, v69, v73
	v_fmac_f32_e32 v100, v9, v11
	v_cvt_f32_i32_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v95, v15, v1 :: v_dual_fmac_f32 v96, v16, v2
	v_mul_f32_e32 v1, v75, v74
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v97, v10, v13 :: v_dual_fmac_f32 v98, v12, v14
	v_dual_mul_f32 v3, v76, v74 :: v_dual_mul_f32 v10, v70, v74
	v_fmac_f32_e32 v93, v1, v2
	v_dual_mul_f32 v2, v72, v74 :: v_dual_mul_f32 v9, v69, v74
	v_cvt_f32_i32_e32 v7, v7
	v_mul_f32_e32 v1, v71, v74
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v114, v28, v30
	v_dual_fmac_f32 v94, v3, v4 :: v_dual_fmac_f32 v91, v2, v6
	v_fmac_f32_e32 v90, v9, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v92, v1, v5 :: v_dual_fmac_f32 v89, v10, v8
	s_xor_b32 s20, s27, -1
	s_mov_b32 s29, 1
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s28, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_17:                               ;   Parent Loop BB2_15 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s20, s29, s24
	s_mov_b32 s31, s21
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s30, s20, 1
	v_add_nc_u32_e32 v83, s26, v158
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[30:31], s[30:31], 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v160, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s31, v161, vcc_lo
	s_mul_u64 s[30:31], s[20:21], s[10:11]
	ds_load_2addr_stride64_b64 v[71:74], v83 offset0:6 offset1:7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	global_load_b128 v[1:4], v[69:70], off
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s29, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s31, 0, s29
	v_add_co_u32 v7, s29, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s31, 0, s29
	s_clause 0x1
	global_load_b64 v[79:80], v[5:6], off offset:40
	global_load_b64 v[81:82], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[5:8], v83 offset1:1
	ds_load_2addr_stride64_b64 v[9:12], v83 offset0:2 offset1:3
	ds_load_2addr_stride64_b64 v[13:16], v83 offset0:4 offset1:5
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v2, 0x88888888, v2
	v_xor_b32_e32 v1, 0x88888888, v1
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v76, 0, v2, s2
	v_cndmask_b32_e64 v75, 0, v1, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v78, 0, v4, s2
	v_cndmask_b32_e64 v77, 0, v3, s2
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[75:76], v[5:6], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[75:76], v[7:8], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[75:76], v[9:10], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[75:76], v[11:12], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[75:76], v[13:14], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[75:76], v[15:16], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[75:76], v[71:72], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[75:76], v[73:74], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[71:74], v83 offset0:32 offset1:96
	v_add_nc_u32_e32 v75, 0x100, v83
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[71:72], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[73:74], v[49:56] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[71:74], v83 offset0:160 offset1:224
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[71:72], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[73:74], v[33:40] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[71:72], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[73:74], v[17:24] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:6 offset1:7
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[71:72], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[73:74], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v80, s0
	v_cndmask_b32_e64 v71, 0, v79, s0
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_b64 v148, v[71:72] offset:8192
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v82, s1
	v_cndmask_b32_e64 v71, 0, v81, s1
	ds_store_b64 v149, v[71:72] offset:8192
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_19
; %bb.18:                               ;   in Loop: Header=BB2_17 Depth=2
	s_add_co_i32 s20, s20, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[71:72], null, s20, s15, v[67:68]
	s_mul_u64 s[30:31], s[20:21], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[75:76], null, 0x48, v159, s[30:31]
	v_mad_co_u64_u32 v[72:73], null, s20, s5, v[72:73]
	v_add_co_u32 v73, s20, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, s31, 0, s20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[71:72], 6, v[71:72]
	v_add_co_u32 v71, vcc_lo, v162, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, v163, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v75, v168
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v76, vcc_lo
	s_clause 0x3
	global_load_u8 v77, v[71:72], off offset:16
	global_load_u8 v78, v[71:72], off offset:32
	global_load_u8 v83, v[71:72], off offset:48
	global_load_u8 v84, v[71:72], off
	v_add_co_u32 v71, s20, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, s31, 0, s20
	s_clause 0x2
	global_load_b64 v[79:80], v[73:74], off offset:8
	global_load_b64 v[81:82], v[71:72], off offset:8
	global_load_b32 v169, v[75:76], off
	s_wait_loadcnt 0x5
	v_lshlrev_b32_e32 v72, 16, v78
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v73, 24, v83
	s_wait_loadcnt 0x3
	v_lshl_or_b32 v71, v84, 8, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_or3_b32 v170, v71, v72, v73
.LBB2_19:                               ; %.preheader421.i
                                        ;   in Loop: Header=BB2_17 Depth=2
	global_load_b128 v[69:72], v[69:70], off offset:512
	v_add_nc_u32_e32 v87, 0, v158
	s_xor_b32 s20, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s19, 0x1400
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[175:178], v87 offset0:22 offset1:23
	s_cselect_b32 s29, s22, 0x1c00
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v70, 0x88888888, v70
	v_xor_b32_e32 v69, 0x88888888, v69
	v_xor_b32_e32 v71, 0x88888888, v71
	v_xor_b32_e32 v72, 0x88888888, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v70, 0, v70, s2
	v_cndmask_b32_e64 v69, 0, v69, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v77, 0, v71, s2
	v_cndmask_b32_e64 v78, 0, v72, s2
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[69:70], v[73:74], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[69:70], v[75:76], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[69:70], v[83:84], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[69:70], v[85:86], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[69:70], v[171:172], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[69:70], v[173:174], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[69:70], v[175:176], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[69:70], v[177:178], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v87, 0x100, v87
	ds_load_2addr_stride64_b64 v[69:72], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:22 offset1:23
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[71:72], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[73:74], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[75:76], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[83:84], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[85:86], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[173:174], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v69, s29, v166
	v_add_nc_u32_e32 v73, s26, v167
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_movk_i32 s26, 0x2000
	ds_load_2addr_b32 v[77:78], v69 offset1:2
	ds_load_2addr_b32 v[75:76], v69 offset0:4 offset1:6
	ds_load_2addr_b32 v[71:72], v69 offset0:8 offset1:10
	ds_load_2addr_b32 v[69:70], v69 offset0:12 offset1:14
	ds_load_2addr_b32 v[87:88], v73 offset1:32
	ds_load_2addr_b32 v[85:86], v73 offset0:64 offset1:96
	ds_load_2addr_b32 v[83:84], v73 offset0:128 offset1:160
	ds_load_2addr_b32 v[73:74], v73 offset0:192 offset1:224
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_16
; %bb.20:                               ; %.preheader422.i
                                        ;   in Loop: Header=BB2_17 Depth=2
	v_lshrrev_b32_e32 v171, v165, v170
	s_and_b32 s20, s28, exec_lo
	s_cselect_b32 s20, s19, 0x1400
	v_cndmask_b32_e64 v80, 0, v80, s0
	v_cndmask_b32_e64 v79, 0, v79, s0
	v_cvt_f32_f16_e64 v171, v171.l
	v_cndmask_b32_e64 v82, 0, v82, s1
	v_cndmask_b32_e64 v81, 0, v81, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v173, s20, v164
	s_cselect_b32 s20, s22, 0x1c00
	v_cndmask_b32_e64 v172, 0, v169, s3
	v_cndmask_b32_e64 v171, 0, v171, s4
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v174, s20, v164
	s_mov_b32 s26, 0
	ds_store_b64 v148, v[79:80]
	ds_store_b64 v149, v[81:82]
	ds_store_b32 v173, v172
	ds_store_b32 v174, v171
	s_branch .LBB2_16
.LBB2_21:                               ; %Flow989
	v_mov_b32_e32 v13, v147
.LBB2_22:                               ; %._crit_edge479.i
	v_mul_u32_u24_e32 v1, 0x500, v145
	v_lshlrev_b32_e32 v2, 2, v146
	v_and_b32_e32 v3, 0x7f, v0
	v_lshrrev_b32_e32 v5, 7, v0
	v_bfe_u32 v0, v0, 4, 3
	s_ashr_i32 s15, s14, 31
	v_add3_u32 v14, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[14:15]
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x500, v0, 0
	v_or_b32_e32 v0, s18, v3
	v_mad_i32_i24 v1, 0x50, v13, v14
	s_ashr_i32 s19, s18, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[18:19], 2
	ds_store_2addr_b32 v1, v156, v157 offset1:20
	ds_store_2addr_b32 v1, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v152, v153 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v150, v151 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v146
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	s_add_co_i32 s6, s18, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mul_lo_u32 v6, s8, v5
	v_add3_u32 v2, v2, v1, v4
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s6, s8
	s_mov_b32 s7, 0x31004000
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s14, 0x80
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s10
	s_mov_b32 s6, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v2
	s_cselect_b32 s1, -1, 0
	s_mov_b32 s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, s1
	v_cmp_gt_i32_e64 s0, s8, v0
	v_add_co_u32 v0, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s17, v8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB2_26
; %bb.23:
	v_or_b32_e32 v7, s14, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_25
; %bb.24:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s13, v7
	v_mad_co_u64_u32 v[7:8], null, s12, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s12, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v1, v8, vcc_lo
	global_load_b32 v9, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v4, v9
	global_store_b32 v[7:8], v9, off
.LBB2_25:                               ; %Flow985
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_26:                               ; %Flow986
	v_add_lshl_u32 v3, v6, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_28
; %bb.27:
	buffer_load_b32 v6, v3, s[4:7], null offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v4, v6
	buffer_store_b32 v4, v3, s[4:7], null offen
.LBB2_28:
	ds_load_b32 v7, v2 offset:8
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, 0, 1, s1
	v_or_b32_e32 v6, 2, v5
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_32
; %bb.29:
	v_or_b32_e32 v8, s14, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v8
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_31
; %bb.30:
	v_ashrrev_i32_e32 v10, 31, v8
	v_mul_lo_u32 v11, s13, v8
	v_mad_co_u64_u32 v[8:9], null, s12, v8, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v10, s12, v10
	v_add3_u32 v9, v9, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v0, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v1, v9, vcc_lo
	global_load_b32 v10, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v7, v10
	global_store_b32 v[8:9], v10, off
.LBB2_31:                               ; %Flow983
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_32:                               ; %Flow984
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_34
; %bb.33:
	s_lshl_b32 s1, s8, 3
	buffer_load_b32 v8, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v7, v7, v8
	buffer_store_b32 v7, v3, s[4:7], s1 offen
.LBB2_34:
	ds_load_b32 v8, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v7, 4, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_38
; %bb.35:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v9, s14, v7
	v_cmp_gt_i32_e32 vcc_lo, s10, v9
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_37
; %bb.36:
	v_ashrrev_i32_e32 v11, 31, v9
	v_mul_lo_u32 v12, s13, v9
	v_mad_co_u64_u32 v[9:10], null, s12, v9, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, s12, v11
	v_add3_u32 v10, v10, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, v0, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc_lo
	global_load_b32 v11, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v8, v11
	global_store_b32 v[9:10], v11, off
.LBB2_37:                               ; %Flow981
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_38:                               ; %Flow982
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_40
; %bb.39:
	s_lshl_b32 s1, s8, 4
	buffer_load_b32 v9, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	buffer_store_b32 v8, v3, s[4:7], s1 offen
.LBB2_40:
	ds_load_b32 v9, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v8, 6, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_44
; %bb.41:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v10, s14, v8
	v_cmp_gt_i32_e32 vcc_lo, s10, v10
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_43
; %bb.42:
	v_ashrrev_i32_e32 v12, 31, v10
	v_mul_lo_u32 v15, s13, v10
	v_mad_co_u64_u32 v[10:11], null, s12, v10, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s12, v12
	v_add3_u32 v11, v11, v12, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, vcc_lo, v0, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v1, v11, vcc_lo
	global_load_b32 v12, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v9, v12
	global_store_b32 v[10:11], v12, off
.LBB2_43:                               ; %Flow979
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_44:                               ; %Flow980
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_46
; %bb.45:
	s_mul_i32 s1, s8, 24
	buffer_load_b32 v10, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	buffer_store_b32 v9, v3, s[4:7], s1 offen
.LBB2_46:
	ds_load_b32 v10, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 8, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_50
; %bb.47:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v11, s14, v9
	v_cmp_gt_i32_e32 vcc_lo, s10, v11
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_49
; %bb.48:
	v_ashrrev_i32_e32 v15, 31, v11
	v_mul_lo_u32 v16, s13, v11
	v_mad_co_u64_u32 v[11:12], null, s12, v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v15, s12, v15
	v_add3_u32 v12, v12, v15, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v1, v12, vcc_lo
	global_load_b32 v15, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v10, v15
	global_store_b32 v[11:12], v15, off
.LBB2_49:                               ; %Flow977
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_50:                               ; %Flow978
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_52
; %bb.51:
	s_lshl_b32 s1, s8, 5
	buffer_load_b32 v11, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v10, v10, v11
	buffer_store_b32 v10, v3, s[4:7], s1 offen
.LBB2_52:
	ds_load_b32 v11, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v10, 10, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_56
; %bb.53:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, s14, v10
	v_cmp_gt_i32_e32 vcc_lo, s10, v12
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_55
; %bb.54:
	v_ashrrev_i32_e32 v17, 31, v12
	v_mul_lo_u32 v18, s13, v12
	v_mad_co_u64_u32 v[15:16], null, s12, v12, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s12, v17
	v_add3_u32 v16, v16, v12, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v12, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v11, v12
	global_store_b32 v[15:16], v12, off
.LBB2_55:                               ; %Flow975
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_56:                               ; %Flow976
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_58
; %bb.57:
	s_mul_i32 s1, s8, 40
	buffer_load_b32 v12, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v11, v11, v12
	buffer_store_b32 v11, v3, s[4:7], s1 offen
.LBB2_58:
	ds_load_b32 v12, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 12, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_62
; %bb.59:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v15, s14, v11
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_61
; %bb.60:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v12, v17
	global_store_b32 v[15:16], v17, off
.LBB2_61:                               ; %Flow973
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_62:                               ; %Flow974
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_64
; %bb.63:
	s_mul_i32 s1, s8, 48
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v12, v12, v15
	buffer_store_b32 v12, v3, s[4:7], s1 offen
.LBB2_64:
	ds_load_b32 v15, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v12, 14, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB2_68
; %bb.65:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v16, s14, v12
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_67
; %bb.66:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s13, v16
	v_mad_co_u64_u32 v[16:17], null, s12, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s12, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v0, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v1, v17, vcc_lo
	global_load_b32 v18, v[16:17], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v18, v15, v18
	global_store_b32 v[16:17], v18, off
.LBB2_67:                               ; %Flow971
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB2_68:                               ; %Flow972
	v_mul_i32_i24_e32 v13, 0x50, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_70
; %bb.69:
	s_mul_i32 s1, s8, 56
	buffer_load_b32 v16, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v15, v15, v16
	buffer_store_b32 v15, v3, s[4:7], s1 offen
.LBB2_70:                               ; %.preheader.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v13, v14, v13
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 16
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v143, v144 offset1:20
	ds_store_2addr_b32 v13, v141, v142 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v139, v140 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v137, v138 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_74
; %bb.71:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_73
; %bb.72:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_73:                               ; %Flow969
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_74:                               ; %Flow970
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_76
; %bb.75:
	s_lshl_b32 s2, s8, 6
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_76:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_80
; %bb.77:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_79
; %bb.78:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_79:                               ; %Flow967
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_80:                               ; %Flow968
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_82
; %bb.81:
	s_mul_i32 s2, s8, 0x48
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_82:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_86
; %bb.83:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_85
; %bb.84:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_85:                               ; %Flow965
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_86:                               ; %Flow966
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_88
; %bb.87:
	s_mul_i32 s2, s8, 0x50
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_88:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_92
; %bb.89:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_91
; %bb.90:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_91:                               ; %Flow963
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_92:                               ; %Flow964
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_94
; %bb.93:
	s_mul_i32 s2, s8, 0x58
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_94:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_98
; %bb.95:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_97
; %bb.96:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_97:                               ; %Flow961
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_98:                               ; %Flow962
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_100
; %bb.99:
	s_mul_i32 s2, s8, 0x60
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_100:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_104
; %bb.101:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_103
; %bb.102:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_103:                              ; %Flow959
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_104:                              ; %Flow960
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_106
; %bb.105:
	s_mul_i32 s2, s8, 0x68
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_106:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_110
; %bb.107:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_109
; %bb.108:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_109:                              ; %Flow957
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_110:                              ; %Flow958
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_112
; %bb.111:
	s_mul_i32 s2, s8, 0x70
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_112:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_116
; %bb.113:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_115
; %bb.114:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_115:                              ; %Flow955
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_116:                              ; %Flow956
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_118
; %bb.117:
	s_mul_i32 s1, s8, 0x78
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_118:                              ; %.preheader.2.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 32
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v135, v136 offset1:20
	ds_store_2addr_b32 v13, v133, v134 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v131, v132 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v129, v130 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_122
; %bb.119:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_121
; %bb.120:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_121:                              ; %Flow953
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_122:                              ; %Flow954
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_124
; %bb.123:
	s_lshl_b32 s2, s8, 7
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_124:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_128
; %bb.125:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_127
; %bb.126:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_127:                              ; %Flow951
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_128:                              ; %Flow952
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_130
; %bb.129:
	s_mul_i32 s2, s8, 0x88
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_130:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_134
; %bb.131:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_133
; %bb.132:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_133:                              ; %Flow949
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_134:                              ; %Flow950
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_136
; %bb.135:
	s_mul_i32 s2, s8, 0x90
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_136:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_140
; %bb.137:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_139
; %bb.138:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_139:                              ; %Flow947
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_140:                              ; %Flow948
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_142
; %bb.141:
	s_mul_i32 s2, s8, 0x98
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_142:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_146
; %bb.143:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_145
; %bb.144:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_145:                              ; %Flow945
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_146:                              ; %Flow946
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_148
; %bb.147:
	s_mul_i32 s2, s8, 0xa0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_148:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_152
; %bb.149:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_151
; %bb.150:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_151:                              ; %Flow943
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_152:                              ; %Flow944
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_154
; %bb.153:
	s_mul_i32 s2, s8, 0xa8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_154:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_158
; %bb.155:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_157
; %bb.156:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_157:                              ; %Flow941
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_158:                              ; %Flow942
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_160
; %bb.159:
	s_mul_i32 s2, s8, 0xb0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_160:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_164
; %bb.161:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_163
; %bb.162:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_163:                              ; %Flow939
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_164:                              ; %Flow940
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_166
; %bb.165:
	s_mul_i32 s1, s8, 0xb8
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_166:                              ; %.preheader.3.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 48
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v127, v128 offset1:20
	ds_store_2addr_b32 v13, v125, v126 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v123, v124 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v121, v122 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_170
; %bb.167:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_169
; %bb.168:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_169:                              ; %Flow937
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_170:                              ; %Flow938
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_172
; %bb.171:
	s_mul_i32 s2, s8, 0xc0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_172:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_176
; %bb.173:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_175
; %bb.174:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_175:                              ; %Flow935
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_176:                              ; %Flow936
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_178
; %bb.177:
	s_mul_i32 s2, s8, 0xc8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_178:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_182
; %bb.179:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_181
; %bb.180:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_181:                              ; %Flow933
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_182:                              ; %Flow934
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_184
; %bb.183:
	s_mul_i32 s2, s8, 0xd0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_184:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_188
; %bb.185:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_187
; %bb.186:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_187:                              ; %Flow931
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_188:                              ; %Flow932
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_190
; %bb.189:
	s_mul_i32 s2, s8, 0xd8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_190:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_194
; %bb.191:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_193
; %bb.192:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_193:                              ; %Flow929
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_194:                              ; %Flow930
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_196
; %bb.195:
	s_mul_i32 s2, s8, 0xe0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_196:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_200
; %bb.197:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_199
; %bb.198:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_199:                              ; %Flow927
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_200:                              ; %Flow928
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_202
; %bb.201:
	s_mul_i32 s2, s8, 0xe8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_202:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_206
; %bb.203:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_205
; %bb.204:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_205:                              ; %Flow925
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_206:                              ; %Flow926
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_208
; %bb.207:
	s_mul_i32 s2, s8, 0xf0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_208:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_212
; %bb.209:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_211
; %bb.210:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_211:                              ; %Flow923
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_212:                              ; %Flow924
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_214
; %bb.213:
	s_mul_i32 s1, s8, 0xf8
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_214:                              ; %.preheader419.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 64
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v119, v120 offset1:20
	ds_store_2addr_b32 v13, v117, v118 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v115, v116 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v113, v114 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_218
; %bb.215:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_217
; %bb.216:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_217:                              ; %Flow921
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_218:                              ; %Flow922
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_220
; %bb.219:
	s_lshl_b32 s2, s8, 8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_220:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_224
; %bb.221:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_223
; %bb.222:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_223:                              ; %Flow919
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_224:                              ; %Flow920
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_226
; %bb.225:
	s_mul_i32 s2, s8, 0x108
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_226:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_230
; %bb.227:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_229
; %bb.228:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_229:                              ; %Flow917
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_230:                              ; %Flow918
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_232
; %bb.231:
	s_mul_i32 s2, s8, 0x110
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_232:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_236
; %bb.233:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_235
; %bb.234:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_235:                              ; %Flow915
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_236:                              ; %Flow916
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_238
; %bb.237:
	s_mul_i32 s2, s8, 0x118
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_238:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_242
; %bb.239:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_241
; %bb.240:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_241:                              ; %Flow913
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_242:                              ; %Flow914
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_244
; %bb.243:
	s_mul_i32 s2, s8, 0x120
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_244:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_248
; %bb.245:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_247
; %bb.246:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_247:                              ; %Flow911
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_248:                              ; %Flow912
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_250
; %bb.249:
	s_mul_i32 s2, s8, 0x128
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_250:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_254
; %bb.251:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_253
; %bb.252:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_253:                              ; %Flow909
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_254:                              ; %Flow910
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_256
; %bb.255:
	s_mul_i32 s2, s8, 0x130
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_256:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_260
; %bb.257:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_259
; %bb.258:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_259:                              ; %Flow907
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_260:                              ; %Flow908
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_262
; %bb.261:
	s_mul_i32 s1, s8, 0x138
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_262:                              ; %.preheader.1.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x50
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v111, v112 offset1:20
	ds_store_2addr_b32 v13, v109, v110 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v107, v108 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v105, v106 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_266
; %bb.263:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_265
; %bb.264:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_265:                              ; %Flow905
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_266:                              ; %Flow906
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_268
; %bb.267:
	s_mul_i32 s2, s8, 0x140
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_268:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_272
; %bb.269:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_271
; %bb.270:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_271:                              ; %Flow903
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_272:                              ; %Flow904
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_274
; %bb.273:
	s_mul_i32 s2, s8, 0x148
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_274:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_278
; %bb.275:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_277
; %bb.276:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_277:                              ; %Flow901
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_278:                              ; %Flow902
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_280
; %bb.279:
	s_mul_i32 s2, s8, 0x150
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_280:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_284
; %bb.281:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_283
; %bb.282:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_283:                              ; %Flow899
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_284:                              ; %Flow900
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_286
; %bb.285:
	s_mul_i32 s2, s8, 0x158
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_286:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_290
; %bb.287:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_289
; %bb.288:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_289:                              ; %Flow897
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_290:                              ; %Flow898
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_292
; %bb.291:
	s_mul_i32 s2, s8, 0x160
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_292:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_296
; %bb.293:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_295
; %bb.294:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_295:                              ; %Flow895
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_296:                              ; %Flow896
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_298
; %bb.297:
	s_mul_i32 s2, s8, 0x168
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_298:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_302
; %bb.299:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_301
; %bb.300:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_301:                              ; %Flow893
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_302:                              ; %Flow894
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_304
; %bb.303:
	s_mul_i32 s2, s8, 0x170
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_304:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_308
; %bb.305:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_307
; %bb.306:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_307:                              ; %Flow891
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_308:                              ; %Flow892
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_310
; %bb.309:
	s_mul_i32 s1, s8, 0x178
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_310:                              ; %.preheader.2.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x60
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v103, v104 offset1:20
	ds_store_2addr_b32 v13, v101, v102 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v99, v100 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v97, v98 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_314
; %bb.311:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_313
; %bb.312:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_313:                              ; %Flow889
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_314:                              ; %Flow890
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_316
; %bb.315:
	s_mul_i32 s2, s8, 0x180
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_316:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_320
; %bb.317:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_319
; %bb.318:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_319:                              ; %Flow887
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_320:                              ; %Flow888
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_322
; %bb.321:
	s_mul_i32 s2, s8, 0x188
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_322:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_326
; %bb.323:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_325
; %bb.324:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_325:                              ; %Flow885
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_326:                              ; %Flow886
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_328
; %bb.327:
	s_mul_i32 s2, s8, 0x190
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_328:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_332
; %bb.329:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_331
; %bb.330:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_331:                              ; %Flow883
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_332:                              ; %Flow884
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_334
; %bb.333:
	s_mul_i32 s2, s8, 0x198
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_334:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_338
; %bb.335:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_337
; %bb.336:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_337:                              ; %Flow881
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_338:                              ; %Flow882
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_340
; %bb.339:
	s_mul_i32 s2, s8, 0x1a0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_340:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_344
; %bb.341:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_343
; %bb.342:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_343:                              ; %Flow879
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_344:                              ; %Flow880
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_346
; %bb.345:
	s_mul_i32 s2, s8, 0x1a8
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_346:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_350
; %bb.347:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_349
; %bb.348:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_349:                              ; %Flow877
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_350:                              ; %Flow878
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_352
; %bb.351:
	s_mul_i32 s2, s8, 0x1b0
	buffer_load_b32 v15, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB2_352:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_356
; %bb.353:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB2_355
; %bb.354:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	global_load_b32 v17, v[15:16], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v17, v14, v17
	global_store_b32 v[15:16], v17, off
.LBB2_355:                              ; %Flow875
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB2_356:                              ; %Flow876
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_358
; %bb.357:
	s_mul_i32 s1, s8, 0x1b8
	buffer_load_b32 v15, v3, s[4:7], s1 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v14, v14, v15
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB2_358:                              ; %.preheader.3.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x70
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v95, v96 offset1:20
	ds_store_2addr_b32 v13, v93, v94 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v92, v91 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v90, v89 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_362
; %bb.359:
	v_or_b32_e32 v5, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v5
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_361
; %bb.360:
	v_ashrrev_i32_e32 v16, 31, v5
	v_mul_lo_u32 v17, s13, v5
	v_mad_co_u64_u32 v[14:15], null, s12, v5, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v5, s12, v16
	v_add3_u32 v15, v15, v5, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v0, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v1, v15, vcc_lo
	global_load_b32 v5, v[14:15], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v13, v5
	global_store_b32 v[14:15], v5, off
.LBB2_361:                              ; %Flow873
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_362:                              ; %Flow874
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_364
; %bb.363:
	s_mul_i32 s2, s8, 0x1c0
	buffer_load_b32 v5, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v13, v5
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_364:
	ds_load_b32 v5, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_368
; %bb.365:
	v_or_b32_e32 v6, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_367
; %bb.366:
	v_ashrrev_i32_e32 v15, 31, v6
	v_mul_lo_u32 v16, s13, v6
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[13:14], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s12, v15
	v_add3_u32 v14, v14, v6, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v1, v14, vcc_lo
	global_load_b32 v6, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v6, v5, v6
	global_store_b32 v[13:14], v6, off
.LBB2_367:                              ; %Flow871
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_368:                              ; %Flow872
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_370
; %bb.369:
	s_mul_i32 s2, s8, 0x1c8
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_370:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_374
; %bb.371:
	v_or_b32_e32 v6, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_373
; %bb.372:
	v_ashrrev_i32_e32 v13, 31, v6
	v_mul_lo_u32 v14, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v13, s12, v13
	v_add3_u32 v7, v7, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	global_load_b32 v13, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v13, v5, v13
	global_store_b32 v[6:7], v13, off
.LBB2_373:                              ; %Flow869
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_374:                              ; %Flow870
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_376
; %bb.375:
	s_mul_i32 s2, s8, 0x1d0
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_376:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_380
; %bb.377:
	v_or_b32_e32 v6, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_379
; %bb.378:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v13, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v5, v8
	global_store_b32 v[6:7], v8, off
.LBB2_379:                              ; %Flow867
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_380:                              ; %Flow868
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_382
; %bb.381:
	s_mul_i32 s2, s8, 0x1d8
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_382:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_386
; %bb.383:
	v_or_b32_e32 v6, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_385
; %bb.384:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v5, v8
	global_store_b32 v[6:7], v8, off
.LBB2_385:                              ; %Flow865
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_386:                              ; %Flow866
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_388
; %bb.387:
	s_mul_i32 s2, s8, 0x1e0
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_388:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_392
; %bb.389:
	v_add_nc_u32_e32 v6, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_391
; %bb.390:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v5, v8
	global_store_b32 v[6:7], v8, off
.LBB2_391:                              ; %Flow863
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_392:                              ; %Flow864
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_394
; %bb.393:
	s_mul_i32 s2, s8, 0x1e8
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_394:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_398
; %bb.395:
	v_add_nc_u32_e32 v6, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB2_397
; %bb.396:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v8, v5, v8
	global_store_b32 v[6:7], v8, off
.LBB2_397:                              ; %Flow861
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB2_398:                              ; %Flow862
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_400
; %bb.399:
	s_mul_i32 s2, s8, 0x1f0
	buffer_load_b32 v6, v3, s[4:7], s2 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB2_400:
	ds_load_b32 v2, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB2_404
; %bb.401:
	v_add_nc_u32_e32 v4, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v4
	s_and_b32 s1, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_403
; %bb.402:
	v_ashrrev_i32_e32 v6, 31, v4
	v_mul_lo_u32 v7, s13, v4
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[4:5], null, s12, v4, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s12, v6
	v_add3_u32 v5, v5, v6, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v1, v5, vcc_lo
	global_load_b32 v4, v[0:1], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v4, v2, v4
	global_store_b32 v[0:1], v4, off
.LBB2_403:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB2_404:                              ; %Flow860
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_406
; %bb.405:
	s_mul_i32 s0, s8, 0x1f8
	buffer_load_b32 v0, v3, s[4:7], s0 offen
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v2, v0
	buffer_store_b32 v0, v3, s[4:7], s0 offen
.LBB2_406:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB2_407:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
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
		.amdhsa_next_free_vgpr 179
		.amdhsa_next_free_sgpr 32
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_vgpr, 179
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.numbered_sgpr, 32
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 19412
; TotalNumSgprs: 34
; NumVgprs: 179
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 179
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
	s_load_b96 s[8:10], s[0:1], 0x18
	s_lshl_b32 s18, ttmp9, 7
	s_lshl_b32 s14, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s18, s8
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s14, s10
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_407
; %bb.1:                                ; %.preheader422.i
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	v_lshrrev_b32_e32 v5, 1, v0
	v_and_b32_e32 v4, 1, v0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_9
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v1, s14, v5
	v_dual_mov_b32 v2, 0 :: v_dual_lshlrev_b32 v3, 2, v4
	v_mov_b32_e32 v7, 0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmpx_gt_i32_e64 s10, v1
	s_cbranch_execz .LBB3_4
; %bb.3:
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[6:7], null, 0x48, v1, s[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, v6, v3
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	global_load_b32 v7, v[6:7], off
.LBB3_4:                                ; %.preheader419.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, s18, v5
	s_add_co_i32 s1, s8, -1
	s_ashr_i32 s3, s8, 31
	s_mov_b32 s2, s8
	s_ashr_i32 s13, s9, 31
	v_min_i32_e32 v8, s1, v6
	s_mov_b32 s12, s9
	v_lshlrev_b32_e32 v9, 3, v5
	s_mul_u64 s[2:3], s[12:13], s[2:3]
	s_mov_b32 s1, exec_lo
	v_ashrrev_i32_e32 v1, 4, v8
	s_lshr_b64 s[2:3], s[2:3], 1
	v_and_b32_e32 v10, 15, v8
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	v_add3_u32 v8, 0, v9, v3
	v_lshlrev_b64_e32 v[1:2], 6, v[1:2]
	s_wait_loadcnt 0x0
	ds_store_b32 v8, v7 offset:4096
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v11, null, s3, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v1, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v11, vcc_lo
                                        ; implicit-def: $vgpr1_lo16
	v_cmpx_ne_u32_e32 0, v4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB3_6
; %bb.5:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off offset:48
	global_load_d16_hi_u8 v1, v[2:3], off offset:32
                                        ; implicit-def: $vgpr2_vgpr3
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB3_6:                                ; %Flow994
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB3_8
; %bb.7:
	s_clause 0x1
	global_load_d16_u8 v1, v[2:3], off
	global_load_d16_hi_u8 v1, v[2:3], off offset:16
	s_wait_loadcnt 0x0
	v_lshlrev_b16 v1.l, 8, v1.l
	s_delay_alu instid0(VALU_DEP_1)
	v_or_b16 v1.l, v1.l, v1.h
.LBB3_8:                                ; %._crit_edge.loopexit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f32_f16_e32 v1, v1.l
	v_cmp_gt_i32_e32 vcc_lo, s8, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	ds_store_b32 v8, v1 offset:6144
.LBB3_9:                                ; %Flow995
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v1, 3, v0
	s_add_co_i32 s3, s10, -1
	v_lshlrev_b32_e32 v10, 4, v0
	v_lshrrev_b32_e32 v145, 5, v0
	v_add_nc_u32_e32 v8, s14, v3
	v_lshlrev_b32_e32 v1, 3, v1
	v_bfe_u32 v11, v0, 1, 1
	v_and_b32_e32 v10, 16, v10
	s_mov_b32 s2, -1
	v_add_nc_u32_e32 v9, 64, v8
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v2, s3, v8
	v_and_or_b32 v12, v145, 6, v11
	v_and_or_b32 v3, v3, 15, v10
	v_or_b32_e32 v10, 8, v145
	v_min_i32_e32 v6, s3, v9
	v_mad_co_u64_u32 v[65:66], null, 0x48, v2, v[1:2]
	v_cmp_gt_i32_e64 s0, s10, v8
	v_lshlrev_b32_e32 v3, 3, v3
	v_and_or_b32 v10, v10, 14, v11
	v_mad_co_u64_u32 v[66:67], null, 0x48, v6, v[1:2]
	v_cmp_gt_i32_e64 s1, s10, v9
	s_cmp_gt_i32 s9, 0xff
	v_lshl_or_b32 v11, v12, 8, v3
	v_lshl_or_b32 v3, v10, 8, v3
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b64 v[1:2], v65, s[6:7] offset:8
	global_load_b64 v[6:7], v66, s[6:7] offset:8
	v_add_nc_u32_e32 v148, 0, v11
	v_add_nc_u32_e32 v149, 0, v3
	s_wait_loadcnt 0x1
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v7, 0, v7, s1
	v_cndmask_b32_e64 v6, 0, v6, s1
	ds_store_b64 v148, v[1:2]
	ds_store_b64 v149, v[6:7]
	v_bfe_u32 v1, v0, 4, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v147, 3, v1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_11
; %bb.10:                               ; %._crit_edge.._crit_edge474_crit_edge.i
	v_lshlrev_b32_e32 v13, 3, v1
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	s_mov_b32 s2, 0
	s_branch .LBB3_12
.LBB3_11:
                                        ; implicit-def: $sgpr12_sgpr13
                                        ; implicit-def: $vgpr13
.LBB3_12:                               ; %Flow991
	v_dual_mov_b32 v89, 0 :: v_dual_and_b32 v146, 15, v0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v123, 0
	v_dual_mov_b32 v124, 0 :: v_dual_mov_b32 v125, 0
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v109, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v113, 0
	v_dual_mov_b32 v114, 0 :: v_dual_mov_b32 v115, 0
	v_dual_mov_b32 v116, 0 :: v_dual_mov_b32 v117, 0
	v_dual_mov_b32 v118, 0 :: v_dual_mov_b32 v119, 0
	v_dual_mov_b32 v120, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v157, 0
	v_mov_b32_e32 v156, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_22
; %bb.13:                               ; %.preheader418.lr.ph.i
	v_dual_mov_b32 v68, 0 :: v_dual_lshlrev_b32 v3, 4, v145
	s_add_co_i32 s12, s8, -16
	s_ashr_i32 s22, s9, 6
	v_and_b32_e32 v6, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v170, v68 :: v_dual_add_nc_u32 v1, s18, v3
	s_ashr_i32 s23, s22, 31
	v_dual_mov_b32 v157, v68 :: v_dual_add_nc_u32 v8, s18, v5
	v_cmp_gt_i32_e64 s2, s8, v1
	s_lshl_b32 s24, s22, 9
	s_lshr_b64 s[22:23], s[22:23], 23
	s_add_co_i32 s19, s8, -1
	v_mov_b32_e32 v169, v68
	v_cndmask_b32_e64 v1, s12, v1, s2
	v_dual_mov_b32 v155, v68 :: v_dual_lshlrev_b32 v158, 3, v6
	v_dual_mov_b32 v153, v68 :: v_dual_lshlrev_b32 v6, 4, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_ashrrev_i32_e32 v1, 4, v1
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v11, s19, v8
	s_ashr_i32 s13, s8, 31
	s_mov_b32 s12, s8
	s_mov_b32 s21, 0
	v_ashrrev_i32_e32 v9, 31, v1
	v_mul_lo_u32 v10, s22, v1
	v_mad_co_u64_u32 v[1:2], null, s24, v1, s[4:5]
	s_mov_b32 s20, s9
	v_dual_mov_b32 v156, v68 :: v_dual_add_nc_u32 v7, s14, v5
	v_mul_lo_u32 v9, s24, v9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[22:23], s[20:21], s[12:13]
	v_mov_b32_e32 v154, v68
	s_wait_alu depctr_sa_sdst(0)
	s_lshr_b64 s[22:23], s[22:23], 1
	v_add_co_u32 v160, vcc_lo, v1, v6
	v_and_b32_e32 v1, 15, v11
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	v_add3_u32 v2, v10, v2, v9
	v_min_i32_e32 v159, s3, v7
	s_ashr_i32 s15, s9, 31
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v162, s3, s22, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v161, null, 0, v2, vcc_lo
	v_dual_mov_b32 v151, v68 :: v_dual_lshlrev_b32 v2, 3, v5
	v_dual_mov_b32 v152, v68 :: v_dual_lshlrev_b32 v5, 2, v4
	v_or_b32_e32 v1, v147, v3
	s_lshr_b32 s15, s15, 24
	v_ashrrev_i32_e32 v67, 4, v11
	v_add_co_ci_u32_e64 v163, null, s23, 0, s3
	v_cmp_gt_i32_e64 s3, s10, v7
	v_add3_u32 v164, 0, v2, v5
	v_cmp_gt_i32_e64 s4, s8, v8
	v_dual_mov_b32 v150, v68 :: v_dual_lshlrev_b32 v165, 4, v4
	v_lshl_add_u32 v166, v1, 3, 0
	v_lshl_add_u32 v167, v146, 3, 0
	v_dual_mov_b32 v119, v68 :: v_dual_lshlrev_b32 v168, 2, v4
	v_dual_mov_b32 v120, v68 :: v_dual_mov_b32 v117, v68
	v_dual_mov_b32 v118, v68 :: v_dual_mov_b32 v115, v68
	v_dual_mov_b32 v116, v68 :: v_dual_mov_b32 v113, v68
	v_dual_mov_b32 v114, v68 :: v_dual_mov_b32 v143, v68
	v_dual_mov_b32 v144, v68 :: v_dual_mov_b32 v141, v68
	v_dual_mov_b32 v142, v68 :: v_dual_mov_b32 v139, v68
	v_dual_mov_b32 v140, v68 :: v_dual_mov_b32 v137, v68
	v_dual_mov_b32 v138, v68 :: v_dual_mov_b32 v111, v68
	v_dual_mov_b32 v112, v68 :: v_dual_mov_b32 v109, v68
	v_dual_mov_b32 v110, v68 :: v_dual_mov_b32 v107, v68
	v_dual_mov_b32 v108, v68 :: v_dual_mov_b32 v105, v68
	v_dual_mov_b32 v106, v68 :: v_dual_mov_b32 v135, v68
	v_dual_mov_b32 v136, v68 :: v_dual_mov_b32 v133, v68
	v_dual_mov_b32 v134, v68 :: v_dual_mov_b32 v131, v68
	v_dual_mov_b32 v132, v68 :: v_dual_mov_b32 v129, v68
	v_dual_mov_b32 v130, v68 :: v_dual_mov_b32 v103, v68
	v_dual_mov_b32 v104, v68 :: v_dual_mov_b32 v101, v68
	v_dual_mov_b32 v102, v68 :: v_dual_mov_b32 v99, v68
	v_dual_mov_b32 v100, v68 :: v_dual_mov_b32 v97, v68
	v_dual_mov_b32 v98, v68 :: v_dual_mov_b32 v127, v68
	v_dual_mov_b32 v128, v68 :: v_dual_mov_b32 v125, v68
	v_dual_mov_b32 v126, v68 :: v_dual_mov_b32 v123, v68
	v_dual_mov_b32 v124, v68 :: v_dual_mov_b32 v121, v68
	v_dual_mov_b32 v122, v68 :: v_dual_mov_b32 v95, v68
	v_dual_mov_b32 v96, v68 :: v_dual_mov_b32 v93, v68
	v_dual_mov_b32 v94, v68 :: v_dual_mov_b32 v91, v68
	v_dual_mov_b32 v92, v68 :: v_dual_mov_b32 v89, v68
	v_mov_b32_e32 v90, v68
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s9, s9, s15
	s_ashr_i32 s15, s8, 4
	s_ashr_i32 s11, s10, 31
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s9, s9, 8
	s_ashr_i32 s5, s15, 31
	s_movk_i32 s19, 0x1000
	s_movk_i32 s22, 0x1800
	s_mov_b32 s26, 0
	s_mov_b32 s23, 0
	s_branch .LBB3_15
.LBB3_14:                               ;   in Loop: Header=BB3_15 Depth=1
	s_and_b32 vcc_lo, exec_lo, s25
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_21
.LBB3_15:                               ; %.preheader418.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_17 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s24, s23, 1
	s_add_co_i32 s23, s23, 1
	s_mov_b32 s28, s21
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s23, s9
	s_mov_b32 s27, -1
	s_cselect_b32 s25, -1, 0
	s_mov_b32 s29, s21
	s_branch .LBB3_17
.LBB3_16:                               ;   in Loop: Header=BB3_17 Depth=2
	s_wait_dscnt 0x3
	v_dual_mul_f32 v79, v77, v87 :: v_dual_mul_f32 v80, v78, v87
	v_cvt_f32_i32_e32 v57, v57
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v61, v61
	v_dual_fmac_f32 v156, v79, v57 :: v_dual_mul_f32 v79, v76, v87
	v_dual_fmac_f32 v157, v80, v58 :: v_dual_mul_f32 v58, v71, v87
	v_mul_f32_e32 v57, v75, v87
	v_cvt_f32_i32_e32 v49, v49
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v155, v79, v60 :: v_dual_mul_f32 v60, v70, v87
	v_fmac_f32_e32 v152, v58, v61
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v154, v57, v59
	v_dual_mul_f32 v57, v72, v87 :: v_dual_mul_f32 v58, v69, v87
	v_cvt_f32_i32_e32 v59, v62
	v_cvt_f32_i32_e32 v61, v63
	v_cvt_f32_i32_e32 v62, v64
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v51, v51
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v153, v57, v59 :: v_dual_fmac_f32 v150, v58, v61
	v_mul_f32_e32 v57, v77, v88
	v_dual_fmac_f32 v151, v60, v62 :: v_dual_mul_f32 v58, v78, v88
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v53, v53
	v_fmac_f32_e32 v143, v57, v49
	v_mul_f32_e32 v49, v75, v88
	v_dual_mul_f32 v57, v76, v88 :: v_dual_fmac_f32 v144, v58, v50
	v_mul_f32_e32 v50, v71, v88
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v142, v57, v52
	v_mul_f32_e32 v52, v70, v88
	v_dual_fmac_f32 v139, v50, v53 :: v_dual_mul_f32 v50, v69, v88
	v_fmac_f32_e32 v141, v49, v51
	v_mul_f32_e32 v49, v72, v88
	v_cvt_f32_i32_e32 v51, v54
	v_cvt_f32_i32_e32 v53, v55
	v_cvt_f32_i32_e32 v54, v56
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_fmac_f32_e32 v137, v50, v53
	s_wait_dscnt 0x2
	v_mul_f32_e32 v50, v78, v85
	v_fmac_f32_e32 v140, v49, v51
	v_dual_mul_f32 v49, v77, v85 :: v_dual_fmac_f32 v138, v52, v54
	v_cvt_f32_i32_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v136, v50, v42
	v_mul_f32_e32 v42, v71, v85
	v_fmac_f32_e32 v135, v49, v41
	v_mul_f32_e32 v49, v76, v85
	v_mul_f32_e32 v41, v75, v85
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_fmac_f32_e32 v134, v49, v44
	v_dual_fmac_f32 v133, v41, v43 :: v_dual_mul_f32 v44, v70, v85
	v_mul_f32_e32 v41, v72, v85
	v_cvt_f32_i32_e32 v43, v46
	v_fmac_f32_e32 v131, v42, v45
	v_mul_f32_e32 v42, v69, v85
	v_cvt_f32_i32_e32 v45, v47
	v_cvt_f32_i32_e32 v46, v48
	v_fmac_f32_e32 v132, v41, v43
	v_mul_f32_e32 v41, v77, v86
	v_cvt_f32_i32_e32 v37, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v129, v42, v45 :: v_dual_fmac_f32 v130, v44, v46
	v_dual_mul_f32 v42, v78, v86 :: v_dual_fmac_f32 v127, v41, v33
	v_mul_f32_e32 v41, v76, v86
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v128, v42, v34
	v_mul_f32_e32 v34, v71, v86
	v_dual_mul_f32 v33, v75, v86 :: v_dual_fmac_f32 v126, v41, v36
	v_mul_f32_e32 v36, v70, v86
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v125, v33, v35
	v_mul_f32_e32 v33, v72, v86
	v_cvt_f32_i32_e32 v35, v38
	v_dual_fmac_f32 v123, v34, v37 :: v_dual_mul_f32 v34, v69, v86
	v_cvt_f32_i32_e32 v37, v39
	v_cvt_f32_i32_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v124, v33, v35
	s_wait_dscnt 0x1
	v_mul_f32_e32 v33, v77, v83
	v_cvt_f32_i32_e32 v29, v29
	v_dual_fmac_f32 v121, v34, v37 :: v_dual_fmac_f32 v122, v36, v38
	v_dual_mul_f32 v34, v78, v83 :: v_dual_fmac_f32 v119, v33, v25
	v_mul_f32_e32 v33, v76, v83
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v120, v34, v26
	v_mul_f32_e32 v26, v71, v83
	v_dual_mul_f32 v25, v75, v83 :: v_dual_fmac_f32 v118, v33, v28
	v_mul_f32_e32 v28, v70, v83
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v117, v25, v27
	v_mul_f32_e32 v25, v72, v83
	v_cvt_f32_i32_e32 v27, v30
	v_cvt_f32_i32_e32 v30, v32
	v_dual_fmac_f32 v115, v26, v29 :: v_dual_mul_f32 v26, v69, v83
	v_cvt_f32_i32_e32 v29, v31
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v116, v25, v27
	v_mul_f32_e32 v25, v77, v84
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v9, v9
	v_fmac_f32_e32 v113, v26, v29
	v_dual_mul_f32 v26, v78, v84 :: v_dual_fmac_f32 v111, v25, v17
	v_mul_f32_e32 v17, v75, v84
	v_mul_f32_e32 v25, v76, v84
	v_cvt_f32_i32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v112, v26, v18
	v_mul_f32_e32 v18, v71, v84
	v_cvt_f32_i32_e32 v11, v11
	v_fmac_f32_e32 v110, v25, v20
	v_mul_f32_e32 v20, v70, v84
	v_cvt_f32_i32_e32 v12, v12
	v_dual_fmac_f32 v107, v18, v21 :: v_dual_mul_f32 v18, v69, v84
	v_fmac_f32_e32 v109, v17, v19
	v_mul_f32_e32 v17, v72, v84
	v_cvt_f32_i32_e32 v19, v22
	v_cvt_f32_i32_e32 v21, v23
	v_cvt_f32_i32_e32 v22, v24
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_fmac_f32_e32 v105, v18, v21
	s_wait_dscnt 0x0
	v_mul_f32_e32 v18, v78, v73
	v_fmac_f32_e32 v108, v17, v19
	v_dual_mul_f32 v17, v77, v73 :: v_dual_fmac_f32 v106, v20, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v104, v18, v10
	v_mul_f32_e32 v10, v71, v73
	v_fmac_f32_e32 v103, v17, v9
	v_mul_f32_e32 v17, v76, v73
	v_mul_f32_e32 v9, v75, v73
	s_barrier_signal -1
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_fmac_f32_e32 v102, v17, v12
	v_dual_fmac_f32 v101, v9, v11 :: v_dual_mul_f32 v12, v70, v73
	v_mul_f32_e32 v9, v72, v73
	v_cvt_f32_i32_e32 v11, v14
	v_fmac_f32_e32 v99, v10, v13
	v_cvt_f32_i32_e32 v13, v15
	v_cvt_f32_i32_e32 v14, v16
	v_dual_mul_f32 v15, v77, v74 :: v_dual_mul_f32 v16, v78, v74
	v_mul_f32_e32 v10, v69, v73
	v_fmac_f32_e32 v100, v9, v11
	v_cvt_f32_i32_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v95, v15, v1 :: v_dual_fmac_f32 v96, v16, v2
	v_mul_f32_e32 v1, v75, v74
	v_cvt_f32_i32_e32 v2, v3
	v_dual_fmac_f32 v97, v10, v13 :: v_dual_fmac_f32 v98, v12, v14
	v_dual_mul_f32 v3, v76, v74 :: v_dual_mul_f32 v10, v70, v74
	v_fmac_f32_e32 v93, v1, v2
	v_dual_mul_f32 v2, v72, v74 :: v_dual_mul_f32 v9, v69, v74
	v_cvt_f32_i32_e32 v7, v7
	v_mul_f32_e32 v1, v71, v74
	v_cvt_f32_i32_e32 v8, v8
	v_fmac_f32_e32 v114, v28, v30
	v_dual_fmac_f32 v94, v3, v4 :: v_dual_fmac_f32 v91, v2, v6
	v_fmac_f32_e32 v90, v9, v7
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v92, v1, v5 :: v_dual_fmac_f32 v89, v10, v8
	s_xor_b32 s20, s27, -1
	s_mov_b32 s29, 1
	s_mov_b32 s27, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s28, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_17:                               ;   Parent Loop BB3_15 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s20, s29, s24
	s_mov_b32 s31, s21
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s30, s20, 1
	v_add_nc_u32_e32 v83, s26, v158
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[30:31], s[30:31], 9
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v160, s30
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s31, v161, vcc_lo
	s_mul_u64 s[30:31], s[20:21], s[10:11]
	ds_load_2addr_stride64_b64 v[71:74], v83 offset0:6 offset1:7
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	global_load_b128 v[1:4], v[69:70], off
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v5, s29, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s31, 0, s29
	v_add_co_u32 v7, s29, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s31, 0, s29
	s_clause 0x1
	global_load_b64 v[79:80], v[5:6], off offset:40
	global_load_b64 v[81:82], v[7:8], off offset:40
	ds_load_2addr_stride64_b64 v[5:8], v83 offset1:1
	ds_load_2addr_stride64_b64 v[9:12], v83 offset0:2 offset1:3
	ds_load_2addr_stride64_b64 v[13:16], v83 offset0:4 offset1:5
	s_wait_loadcnt 0x2
	v_xor_b32_e32 v2, 0x88888888, v2
	v_xor_b32_e32 v1, 0x88888888, v1
	v_xor_b32_e32 v4, 0x88888888, v4
	v_xor_b32_e32 v3, 0x88888888, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v76, 0, v2, s2
	v_cndmask_b32_e64 v75, 0, v1, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v78, 0, v4, s2
	v_cndmask_b32_e64 v77, 0, v3, s2
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[75:76], v[5:6], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[75:76], v[7:8], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[75:76], v[9:10], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[75:76], v[11:12], 0 neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[75:76], v[13:14], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[75:76], v[15:16], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[75:76], v[71:72], 0 neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[75:76], v[73:74], 0 neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	ds_load_2addr_b64 v[71:74], v83 offset0:32 offset1:96
	v_add_nc_u32_e32 v75, 0x100, v83
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[71:72], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[73:74], v[49:56] neg_lo:[1,1,0]
	ds_load_2addr_b64 v[71:74], v83 offset0:160 offset1:224
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[71:72], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[73:74], v[33:40] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[71:72], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[73:74], v[17:24] neg_lo:[1,1,0]
	ds_load_2addr_stride64_b64 v[71:74], v75 offset0:6 offset1:7
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[71:72], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[73:74], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v80, s0
	v_cndmask_b32_e64 v71, 0, v79, s0
	s_and_b32 s26, s28, s25
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s26
	ds_store_b64 v148, v[71:72] offset:8192
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v72, 0, v82, s1
	v_cndmask_b32_e64 v71, 0, v81, s1
	ds_store_b64 v149, v[71:72] offset:8192
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_19
; %bb.18:                               ;   in Loop: Header=BB3_17 Depth=2
	s_add_co_i32 s20, s20, 1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[71:72], null, s20, s15, v[67:68]
	s_mul_u64 s[30:31], s[20:21], s[10:11]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[30:31], s[30:31], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_i64_i32 v[75:76], null, 0x48, v159, s[30:31]
	v_mad_co_u64_u32 v[72:73], null, s20, s5, v[72:73]
	v_add_co_u32 v73, s20, s30, v65
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v74, null, s31, 0, s20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[71:72], 6, v[71:72]
	v_add_co_u32 v71, vcc_lo, v162, v71
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v72, null, v163, v72, vcc_lo
	v_add_co_u32 v75, vcc_lo, v75, v168
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v76, null, 0, v76, vcc_lo
	s_clause 0x3
	global_load_u8 v77, v[71:72], off offset:16
	global_load_u8 v78, v[71:72], off offset:32
	global_load_u8 v83, v[71:72], off offset:48
	global_load_u8 v84, v[71:72], off
	v_add_co_u32 v71, s20, s30, v66
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v72, null, s31, 0, s20
	s_clause 0x2
	global_load_b64 v[79:80], v[73:74], off offset:8
	global_load_b64 v[81:82], v[71:72], off offset:8
	global_load_b32 v169, v[75:76], off
	s_wait_loadcnt 0x5
	v_lshlrev_b32_e32 v72, 16, v78
	s_wait_loadcnt 0x4
	v_lshlrev_b32_e32 v73, 24, v83
	s_wait_loadcnt 0x3
	v_lshl_or_b32 v71, v84, 8, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_or3_b32 v170, v71, v72, v73
.LBB3_19:                               ; %.preheader416.i
                                        ;   in Loop: Header=BB3_17 Depth=2
	global_load_b128 v[69:72], v[69:70], off offset:512
	v_add_nc_u32_e32 v87, 0, v158
	s_xor_b32 s20, s26, -1
	s_and_b32 s26, s27, exec_lo
	s_cselect_b32 s26, s19, 0x1400
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[175:178], v87 offset0:22 offset1:23
	s_cselect_b32 s29, s22, 0x1c00
	s_wait_loadcnt 0x0
	v_xor_b32_e32 v70, 0x88888888, v70
	v_xor_b32_e32 v69, 0x88888888, v69
	v_xor_b32_e32 v71, 0x88888888, v71
	v_xor_b32_e32 v72, 0x88888888, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v70, 0, v70, s2
	v_cndmask_b32_e64 v69, 0, v69, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v77, 0, v71, s2
	v_cndmask_b32_e64 v78, 0, v72, s2
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3)
	v_wmma_i32_16x16x32_iu4 v[57:64], v[69:70], v[73:74], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[69:70], v[75:76], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[69:70], v[83:84], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[69:70], v[85:86], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[69:70], v[171:172], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[69:70], v[173:174], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[69:70], v[175:176], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[69:70], v[177:178], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	v_add_nc_u32_e32 v87, 0x100, v87
	ds_load_2addr_stride64_b64 v[69:72], v87 offset0:16 offset1:17
	ds_load_2addr_stride64_b64 v[73:76], v87 offset0:18 offset1:19
	ds_load_2addr_stride64_b64 v[83:86], v87 offset0:20 offset1:21
	ds_load_2addr_stride64_b64 v[171:174], v87 offset0:22 offset1:23
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[57:64], v[77:78], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[77:78], v[71:72], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[41:48], v[77:78], v[73:74], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[77:78], v[75:76], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[77:78], v[83:84], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[77:78], v[85:86], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[9:16], v[77:78], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[77:78], v[173:174], v[1:8] neg_lo:[1,1,0]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v69, s29, v166
	v_add_nc_u32_e32 v73, s26, v167
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_movk_i32 s26, 0x2000
	ds_load_2addr_b32 v[77:78], v69 offset1:2
	ds_load_2addr_b32 v[75:76], v69 offset0:4 offset1:6
	ds_load_2addr_b32 v[71:72], v69 offset0:8 offset1:10
	ds_load_2addr_b32 v[69:70], v69 offset0:12 offset1:14
	ds_load_2addr_b32 v[87:88], v73 offset1:32
	ds_load_2addr_b32 v[85:86], v73 offset0:64 offset1:96
	ds_load_2addr_b32 v[83:84], v73 offset0:128 offset1:160
	ds_load_2addr_b32 v[73:74], v73 offset0:192 offset1:224
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_16
; %bb.20:                               ; %.preheader417.i
                                        ;   in Loop: Header=BB3_17 Depth=2
	v_lshrrev_b32_e32 v171, v165, v170
	s_and_b32 s20, s28, exec_lo
	s_cselect_b32 s20, s19, 0x1400
	v_cndmask_b32_e64 v80, 0, v80, s0
	v_cndmask_b32_e64 v79, 0, v79, s0
	v_cvt_f32_f16_e64 v171, v171.l
	v_cndmask_b32_e64 v82, 0, v82, s1
	v_cndmask_b32_e64 v81, 0, v81, s1
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v173, s20, v164
	s_cselect_b32 s20, s22, 0x1c00
	v_cndmask_b32_e64 v172, 0, v169, s3
	v_cndmask_b32_e64 v171, 0, v171, s4
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v174, s20, v164
	s_mov_b32 s26, 0
	ds_store_b64 v148, v[79:80]
	ds_store_b64 v149, v[81:82]
	ds_store_b32 v173, v172
	ds_store_b32 v174, v171
	s_branch .LBB3_16
.LBB3_21:                               ; %Flow989
	v_mov_b32_e32 v13, v147
.LBB3_22:                               ; %._crit_edge474.i
	v_mul_u32_u24_e32 v1, 0x500, v145
	v_lshlrev_b32_e32 v2, 2, v146
	v_and_b32_e32 v3, 0x7f, v0
	v_lshrrev_b32_e32 v5, 7, v0
	v_bfe_u32 v0, v0, 4, 3
	s_ashr_i32 s15, s14, 31
	v_add3_u32 v14, 0, v1, v2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[0:1], s[12:13], s[14:15]
	v_lshlrev_b32_e32 v4, 2, v5
	v_mad_u32_u24 v2, 0x500, v0, 0
	v_or_b32_e32 v0, s18, v3
	v_mad_i32_i24 v1, 0x50, v13, v14
	s_ashr_i32 s19, s18, 31
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b64 s[2:3], s[18:19], 2
	ds_store_2addr_b32 v1, v156, v157 offset1:20
	ds_store_2addr_b32 v1, v154, v155 offset0:40 offset1:60
	ds_store_2addr_b32 v1, v152, v153 offset0:80 offset1:100
	ds_store_2addr_b32 v1, v150, v151 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_mul_u32_u24_e32 v1, 0x50, v146
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[16:17], s[0:1]
	s_add_co_i32 s6, s18, 0x80
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[0:1], s[2:3]
	v_mul_lo_u32 v6, s8, v5
	v_add3_u32 v2, v2, v1, v4
	v_ashrrev_i32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s5, s5, 0xffff
	s_cmp_gt_i32 s6, s8
	s_mov_b32 s7, 0x31004000
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s14, 0x80
	v_lshlrev_b64_e32 v[7:8], 2, v[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_gt_i32 s1, s10
	s_mov_b32 s6, -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v4, v2
	s_cselect_b32 s1, -1, 0
	s_mov_b32 s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s1, s0, s1
	v_cmp_gt_i32_e64 s0, s8, v0
	v_add_co_u32 v0, vcc_lo, s16, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s17, v8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB3_26
; %bb.23:
	v_or_b32_e32 v7, s14, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v7
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_25
; %bb.24:
	v_ashrrev_i32_e32 v9, 31, v7
	v_mul_lo_u32 v10, s13, v7
	v_mad_co_u64_u32 v[7:8], null, s12, v7, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v9, s12, v9
	v_add3_u32 v8, v8, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v1, v8, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[7:8], v4, off
.LBB3_25:                               ; %Flow985
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_26:                               ; %Flow986
	v_add_lshl_u32 v3, v6, v3, 2
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_28
; %bb.27:
	s_wait_dscnt 0x0
	buffer_store_b32 v4, v3, s[4:7], null offen
.LBB3_28:
	ds_load_b32 v7, v2 offset:8
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v4, 0, 1, s1
	v_or_b32_e32 v6, 2, v5
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_32
; %bb.29:
	v_or_b32_e32 v8, s14, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v8
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_31
; %bb.30:
	v_ashrrev_i32_e32 v10, 31, v8
	v_mul_lo_u32 v11, s13, v8
	v_mad_co_u64_u32 v[8:9], null, s12, v8, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v10, s12, v10
	v_add3_u32 v9, v9, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v8, vcc_lo, v0, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v1, v9, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[8:9], v7, off
.LBB3_31:                               ; %Flow983
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_32:                               ; %Flow984
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_34
; %bb.33:
	s_lshl_b32 s1, s8, 3
	s_wait_dscnt 0x0
	buffer_store_b32 v7, v3, s[4:7], s1 offen
.LBB3_34:
	ds_load_b32 v8, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v7, 4, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_38
; %bb.35:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v9, s14, v7
	v_cmp_gt_i32_e32 vcc_lo, s10, v9
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_37
; %bb.36:
	v_ashrrev_i32_e32 v11, 31, v9
	v_mul_lo_u32 v12, s13, v9
	v_mad_co_u64_u32 v[9:10], null, s12, v9, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v11, s12, v11
	v_add3_u32 v10, v10, v11, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v9, vcc_lo, v0, v9
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v1, v10, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[9:10], v8, off
.LBB3_37:                               ; %Flow981
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_38:                               ; %Flow982
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_40
; %bb.39:
	s_lshl_b32 s1, s8, 4
	s_wait_dscnt 0x0
	buffer_store_b32 v8, v3, s[4:7], s1 offen
.LBB3_40:
	ds_load_b32 v9, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v8, 6, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_44
; %bb.41:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v10, s14, v8
	v_cmp_gt_i32_e32 vcc_lo, s10, v10
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_43
; %bb.42:
	v_ashrrev_i32_e32 v12, 31, v10
	v_mul_lo_u32 v15, s13, v10
	v_mad_co_u64_u32 v[10:11], null, s12, v10, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s12, v12
	v_add3_u32 v11, v11, v12, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v10, vcc_lo, v0, v10
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v1, v11, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[10:11], v9, off
.LBB3_43:                               ; %Flow979
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_44:                               ; %Flow980
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_46
; %bb.45:
	s_mul_i32 s1, s8, 24
	s_wait_dscnt 0x0
	buffer_store_b32 v9, v3, s[4:7], s1 offen
.LBB3_46:
	ds_load_b32 v10, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v9, 8, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_50
; %bb.47:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v11, s14, v9
	v_cmp_gt_i32_e32 vcc_lo, s10, v11
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_49
; %bb.48:
	v_ashrrev_i32_e32 v15, 31, v11
	v_mul_lo_u32 v16, s13, v11
	v_mad_co_u64_u32 v[11:12], null, s12, v11, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v15, s12, v15
	v_add3_u32 v12, v12, v15, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v11, vcc_lo, v0, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v1, v12, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[11:12], v10, off
.LBB3_49:                               ; %Flow977
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_50:                               ; %Flow978
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_52
; %bb.51:
	s_lshl_b32 s1, s8, 5
	s_wait_dscnt 0x0
	buffer_store_b32 v10, v3, s[4:7], s1 offen
.LBB3_52:
	ds_load_b32 v11, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v10, 10, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_56
; %bb.53:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v12, s14, v10
	v_cmp_gt_i32_e32 vcc_lo, s10, v12
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_55
; %bb.54:
	v_ashrrev_i32_e32 v17, 31, v12
	v_mul_lo_u32 v18, s13, v12
	v_mad_co_u64_u32 v[15:16], null, s12, v12, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v12, s12, v17
	v_add3_u32 v16, v16, v12, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v11, off
.LBB3_55:                               ; %Flow975
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_56:                               ; %Flow976
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_58
; %bb.57:
	s_mul_i32 s1, s8, 40
	s_wait_dscnt 0x0
	buffer_store_b32 v11, v3, s[4:7], s1 offen
.LBB3_58:
	ds_load_b32 v12, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v11, 12, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_62
; %bb.59:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v15, s14, v11
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_61
; %bb.60:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v12, off
.LBB3_61:                               ; %Flow973
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_62:                               ; %Flow974
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_64
; %bb.63:
	s_mul_i32 s1, s8, 48
	s_wait_dscnt 0x0
	buffer_store_b32 v12, v3, s[4:7], s1 offen
.LBB3_64:
	ds_load_b32 v15, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_wait_dscnt 0x1
	v_or_b32_e32 v12, 14, v5
	s_mov_b32 s1, -1
	s_cbranch_vccnz .LBB3_68
; %bb.65:
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_or_b32_e32 v16, s14, v12
	v_cmp_gt_i32_e32 vcc_lo, s10, v16
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_67
; %bb.66:
	v_ashrrev_i32_e32 v18, 31, v16
	v_mul_lo_u32 v19, s13, v16
	v_mad_co_u64_u32 v[16:17], null, s12, v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v18, s12, v18
	v_add3_u32 v17, v17, v18, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[16:17], 2, v[16:17]
	v_add_co_u32 v16, vcc_lo, v0, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, v1, v17, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[16:17], v15, off
.LBB3_67:                               ; %Flow971
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s1, 0
.LBB3_68:                               ; %Flow972
	v_mul_i32_i24_e32 v13, 0x50, v13
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_70
; %bb.69:
	s_mul_i32 s1, s8, 56
	s_wait_dscnt 0x0
	buffer_store_b32 v15, v3, s[4:7], s1 offen
.LBB3_70:                               ; %.preheader.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v13, v14, v13
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 16
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v143, v144 offset1:20
	ds_store_2addr_b32 v13, v141, v142 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v139, v140 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v137, v138 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_74
; %bb.71:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_73
; %bb.72:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_73:                               ; %Flow969
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_74:                               ; %Flow970
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_76
; %bb.75:
	s_lshl_b32 s2, s8, 6
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_76:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_80
; %bb.77:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_79
; %bb.78:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_79:                               ; %Flow967
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_80:                               ; %Flow968
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_82
; %bb.81:
	s_mul_i32 s2, s8, 0x48
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_82:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_86
; %bb.83:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_85
; %bb.84:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_85:                               ; %Flow965
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_86:                               ; %Flow966
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_88
; %bb.87:
	s_mul_i32 s2, s8, 0x50
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_88:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_92
; %bb.89:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_91
; %bb.90:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_91:                               ; %Flow963
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_92:                               ; %Flow964
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_94
; %bb.93:
	s_mul_i32 s2, s8, 0x58
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_94:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_98
; %bb.95:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_97
; %bb.96:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_97:                               ; %Flow961
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_98:                               ; %Flow962
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_100
; %bb.99:
	s_mul_i32 s2, s8, 0x60
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_100:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_104
; %bb.101:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_103
; %bb.102:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_103:                              ; %Flow959
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_104:                              ; %Flow960
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_106
; %bb.105:
	s_mul_i32 s2, s8, 0x68
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_106:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_110
; %bb.107:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_109
; %bb.108:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_109:                              ; %Flow957
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_110:                              ; %Flow958
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_112
; %bb.111:
	s_mul_i32 s2, s8, 0x70
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_112:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_116
; %bb.113:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_115
; %bb.114:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_115:                              ; %Flow955
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_116:                              ; %Flow956
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_118
; %bb.117:
	s_mul_i32 s1, s8, 0x78
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_118:                              ; %.preheader.2.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 32
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v135, v136 offset1:20
	ds_store_2addr_b32 v13, v133, v134 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v131, v132 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v129, v130 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_122
; %bb.119:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_121
; %bb.120:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_121:                              ; %Flow953
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_122:                              ; %Flow954
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_124
; %bb.123:
	s_lshl_b32 s2, s8, 7
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_124:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_128
; %bb.125:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_127
; %bb.126:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_127:                              ; %Flow951
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_128:                              ; %Flow952
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_130
; %bb.129:
	s_mul_i32 s2, s8, 0x88
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_130:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_134
; %bb.131:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_133
; %bb.132:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_133:                              ; %Flow949
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_134:                              ; %Flow950
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_136
; %bb.135:
	s_mul_i32 s2, s8, 0x90
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_136:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_140
; %bb.137:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_139
; %bb.138:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_139:                              ; %Flow947
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_140:                              ; %Flow948
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_142
; %bb.141:
	s_mul_i32 s2, s8, 0x98
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_142:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_146
; %bb.143:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_145
; %bb.144:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_145:                              ; %Flow945
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_146:                              ; %Flow946
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_148
; %bb.147:
	s_mul_i32 s2, s8, 0xa0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_148:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_152
; %bb.149:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_151
; %bb.150:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_151:                              ; %Flow943
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_152:                              ; %Flow944
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_154
; %bb.153:
	s_mul_i32 s2, s8, 0xa8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_154:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_158
; %bb.155:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_157
; %bb.156:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_157:                              ; %Flow941
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_158:                              ; %Flow942
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_160
; %bb.159:
	s_mul_i32 s2, s8, 0xb0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_160:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_164
; %bb.161:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_163
; %bb.162:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_163:                              ; %Flow939
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_164:                              ; %Flow940
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_166
; %bb.165:
	s_mul_i32 s1, s8, 0xb8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_166:                              ; %.preheader.3.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 48
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v127, v128 offset1:20
	ds_store_2addr_b32 v13, v125, v126 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v123, v124 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v121, v122 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_170
; %bb.167:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_169
; %bb.168:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_169:                              ; %Flow937
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_170:                              ; %Flow938
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_172
; %bb.171:
	s_mul_i32 s2, s8, 0xc0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_172:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_176
; %bb.173:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_175
; %bb.174:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_175:                              ; %Flow935
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_176:                              ; %Flow936
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_178
; %bb.177:
	s_mul_i32 s2, s8, 0xc8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_178:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_182
; %bb.179:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_181
; %bb.180:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_181:                              ; %Flow933
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_182:                              ; %Flow934
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_184
; %bb.183:
	s_mul_i32 s2, s8, 0xd0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_184:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_188
; %bb.185:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_187
; %bb.186:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_187:                              ; %Flow931
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_188:                              ; %Flow932
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_190
; %bb.189:
	s_mul_i32 s2, s8, 0xd8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_190:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_194
; %bb.191:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_193
; %bb.192:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_193:                              ; %Flow929
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_194:                              ; %Flow930
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_196
; %bb.195:
	s_mul_i32 s2, s8, 0xe0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_196:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_200
; %bb.197:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_199
; %bb.198:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_199:                              ; %Flow927
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_200:                              ; %Flow928
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_202
; %bb.201:
	s_mul_i32 s2, s8, 0xe8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_202:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_206
; %bb.203:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_205
; %bb.204:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_205:                              ; %Flow925
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_206:                              ; %Flow926
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_208
; %bb.207:
	s_mul_i32 s2, s8, 0xf0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_208:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_212
; %bb.209:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_211
; %bb.210:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_211:                              ; %Flow923
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_212:                              ; %Flow924
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_214
; %bb.213:
	s_mul_i32 s1, s8, 0xf8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_214:                              ; %.preheader414.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 64
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v119, v120 offset1:20
	ds_store_2addr_b32 v13, v117, v118 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v115, v116 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v113, v114 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_218
; %bb.215:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_217
; %bb.216:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_217:                              ; %Flow921
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_218:                              ; %Flow922
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_220
; %bb.219:
	s_lshl_b32 s2, s8, 8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_220:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_224
; %bb.221:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_223
; %bb.222:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_223:                              ; %Flow919
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_224:                              ; %Flow920
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_226
; %bb.225:
	s_mul_i32 s2, s8, 0x108
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_226:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_230
; %bb.227:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_229
; %bb.228:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_229:                              ; %Flow917
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_230:                              ; %Flow918
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_232
; %bb.231:
	s_mul_i32 s2, s8, 0x110
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_232:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_236
; %bb.233:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_235
; %bb.234:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_235:                              ; %Flow915
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_236:                              ; %Flow916
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_238
; %bb.237:
	s_mul_i32 s2, s8, 0x118
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_238:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_242
; %bb.239:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_241
; %bb.240:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_241:                              ; %Flow913
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_242:                              ; %Flow914
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_244
; %bb.243:
	s_mul_i32 s2, s8, 0x120
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_244:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_248
; %bb.245:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_247
; %bb.246:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_247:                              ; %Flow911
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_248:                              ; %Flow912
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_250
; %bb.249:
	s_mul_i32 s2, s8, 0x128
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_250:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_254
; %bb.251:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_253
; %bb.252:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_253:                              ; %Flow909
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_254:                              ; %Flow910
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_256
; %bb.255:
	s_mul_i32 s2, s8, 0x130
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_256:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_260
; %bb.257:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_259
; %bb.258:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_259:                              ; %Flow907
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_260:                              ; %Flow908
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_262
; %bb.261:
	s_mul_i32 s1, s8, 0x138
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_262:                              ; %.preheader.1.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x50
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v111, v112 offset1:20
	ds_store_2addr_b32 v13, v109, v110 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v107, v108 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v105, v106 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_266
; %bb.263:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_265
; %bb.264:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_265:                              ; %Flow905
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_266:                              ; %Flow906
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_268
; %bb.267:
	s_mul_i32 s2, s8, 0x140
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_268:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_272
; %bb.269:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_271
; %bb.270:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_271:                              ; %Flow903
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_272:                              ; %Flow904
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_274
; %bb.273:
	s_mul_i32 s2, s8, 0x148
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_274:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_278
; %bb.275:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_277
; %bb.276:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_277:                              ; %Flow901
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_278:                              ; %Flow902
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_280
; %bb.279:
	s_mul_i32 s2, s8, 0x150
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_280:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_284
; %bb.281:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_283
; %bb.282:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_283:                              ; %Flow899
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_284:                              ; %Flow900
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_286
; %bb.285:
	s_mul_i32 s2, s8, 0x158
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_286:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_290
; %bb.287:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_289
; %bb.288:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_289:                              ; %Flow897
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_290:                              ; %Flow898
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_292
; %bb.291:
	s_mul_i32 s2, s8, 0x160
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_292:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_296
; %bb.293:
	v_add_nc_u32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_295
; %bb.294:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_295:                              ; %Flow895
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_296:                              ; %Flow896
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_298
; %bb.297:
	s_mul_i32 s2, s8, 0x168
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_298:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_302
; %bb.299:
	v_add_nc_u32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_301
; %bb.300:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_301:                              ; %Flow893
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_302:                              ; %Flow894
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_304
; %bb.303:
	s_mul_i32 s2, s8, 0x170
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_304:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_308
; %bb.305:
	v_add_nc_u32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_307
; %bb.306:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_307:                              ; %Flow891
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_308:                              ; %Flow892
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_310
; %bb.309:
	s_mul_i32 s1, s8, 0x178
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_310:                              ; %.preheader.2.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x60
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v103, v104 offset1:20
	ds_store_2addr_b32 v13, v101, v102 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v99, v100 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v97, v98 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v14, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_314
; %bb.311:
	v_or_b32_e32 v15, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_313
; %bb.312:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_313:                              ; %Flow889
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_314:                              ; %Flow890
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_316
; %bb.315:
	s_mul_i32 s2, s8, 0x180
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_316:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_320
; %bb.317:
	v_or_b32_e32 v15, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_319
; %bb.318:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_319:                              ; %Flow887
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_320:                              ; %Flow888
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_322
; %bb.321:
	s_mul_i32 s2, s8, 0x188
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_322:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_326
; %bb.323:
	v_or_b32_e32 v15, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_325
; %bb.324:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_325:                              ; %Flow885
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_326:                              ; %Flow886
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_328
; %bb.327:
	s_mul_i32 s2, s8, 0x190
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_328:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_332
; %bb.329:
	v_or_b32_e32 v15, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_331
; %bb.330:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_331:                              ; %Flow883
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_332:                              ; %Flow884
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_334
; %bb.333:
	s_mul_i32 s2, s8, 0x198
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_334:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_338
; %bb.335:
	v_or_b32_e32 v15, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_337
; %bb.336:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_337:                              ; %Flow881
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_338:                              ; %Flow882
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_340
; %bb.339:
	s_mul_i32 s2, s8, 0x1a0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_340:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_344
; %bb.341:
	v_or_b32_e32 v15, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_343
; %bb.342:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_343:                              ; %Flow879
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_344:                              ; %Flow880
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_346
; %bb.345:
	s_mul_i32 s2, s8, 0x1a8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_346:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_350
; %bb.347:
	v_or_b32_e32 v15, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_349
; %bb.348:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_349:                              ; %Flow877
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_350:                              ; %Flow878
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_352
; %bb.351:
	s_mul_i32 s2, s8, 0x1b0
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s2 offen
.LBB3_352:
	s_wait_dscnt 0x0
	ds_load_b32 v14, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_356
; %bb.353:
	v_or_b32_e32 v15, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v15
	s_and_b32 s2, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB3_355
; %bb.354:
	v_ashrrev_i32_e32 v17, 31, v15
	v_mul_lo_u32 v18, s13, v15
	v_mad_co_u64_u32 v[15:16], null, s12, v15, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v17, s12, v17
	v_add3_u32 v16, v16, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[15:16], 2, v[15:16]
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v16, null, v1, v16, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[15:16], v14, off
.LBB3_355:                              ; %Flow875
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_mov_b32 s2, 0
.LBB3_356:                              ; %Flow876
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_358
; %bb.357:
	s_mul_i32 s1, s8, 0x1b8
	s_wait_dscnt 0x0
	buffer_store_b32 v14, v3, s[4:7], s1 offen
.LBB3_358:                              ; %.preheader.3.1.i
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_or_b32 s1, s14, 0x70
	s_mov_b32 s2, -1
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v13, v95, v96 offset1:20
	ds_store_2addr_b32 v13, v93, v94 offset0:40 offset1:60
	ds_store_2addr_b32 v13, v92, v91 offset0:80 offset1:100
	ds_store_2addr_b32 v13, v90, v89 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_load_b32 v13, v2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_362
; %bb.359:
	v_or_b32_e32 v5, s1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v5
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_361
; %bb.360:
	v_ashrrev_i32_e32 v16, 31, v5
	v_mul_lo_u32 v17, s13, v5
	v_mad_co_u64_u32 v[14:15], null, s12, v5, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v5, s12, v16
	v_add3_u32 v15, v15, v5, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[14:15], 2, v[14:15]
	v_add_co_u32 v14, vcc_lo, v0, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v15, null, v1, v15, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[14:15], v13, off
.LBB3_361:                              ; %Flow873
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_362:                              ; %Flow874
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_364
; %bb.363:
	s_mul_i32 s2, s8, 0x1c0
	s_wait_dscnt 0x0
	buffer_store_b32 v13, v3, s[4:7], s2 offen
.LBB3_364:
	ds_load_b32 v5, v2 offset:8
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_368
; %bb.365:
	v_or_b32_e32 v6, s1, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_367
; %bb.366:
	v_ashrrev_i32_e32 v15, 31, v6
	v_mul_lo_u32 v16, s13, v6
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[13:14], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s12, v15
	v_add3_u32 v14, v14, v6, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v13, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v1, v14, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[13:14], v5, off
.LBB3_367:                              ; %Flow871
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_368:                              ; %Flow872
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_370
; %bb.369:
	s_mul_i32 s2, s8, 0x1c8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_370:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:16
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_374
; %bb.371:
	v_or_b32_e32 v6, s1, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_373
; %bb.372:
	v_ashrrev_i32_e32 v13, 31, v6
	v_mul_lo_u32 v14, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v13, s12, v13
	v_add3_u32 v7, v7, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB3_373:                              ; %Flow869
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_374:                              ; %Flow870
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_376
; %bb.375:
	s_mul_i32 s2, s8, 0x1d0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_376:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:24
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_380
; %bb.377:
	v_or_b32_e32 v6, s1, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_379
; %bb.378:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v13, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB3_379:                              ; %Flow867
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_380:                              ; %Flow868
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_382
; %bb.381:
	s_mul_i32 s2, s8, 0x1d8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_382:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:32
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_386
; %bb.383:
	v_or_b32_e32 v6, s1, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_385
; %bb.384:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB3_385:                              ; %Flow865
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_386:                              ; %Flow866
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_388
; %bb.387:
	s_mul_i32 s2, s8, 0x1e0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_388:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:40
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_392
; %bb.389:
	v_add_nc_u32_e32 v6, s1, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_391
; %bb.390:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB3_391:                              ; %Flow863
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_392:                              ; %Flow864
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_394
; %bb.393:
	s_mul_i32 s2, s8, 0x1e8
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_394:
	s_wait_dscnt 0x0
	ds_load_b32 v5, v2 offset:48
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_398
; %bb.395:
	v_add_nc_u32_e32 v6, s1, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v6
	s_and_b32 s3, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s3
	s_cbranch_execz .LBB3_397
; %bb.396:
	v_ashrrev_i32_e32 v8, 31, v6
	v_mul_lo_u32 v9, s13, v6
	v_mad_co_u64_u32 v[6:7], null, s12, v6, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v8, s12, v8
	v_add3_u32 v7, v7, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v6, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v1, v7, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[6:7], v5, off
.LBB3_397:                              ; %Flow861
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s2, 0
.LBB3_398:                              ; %Flow862
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_400
; %bb.399:
	s_mul_i32 s2, s8, 0x1f0
	s_wait_dscnt 0x0
	buffer_store_b32 v5, v3, s[4:7], s2 offen
.LBB3_400:
	ds_load_b32 v2, v2 offset:56
	v_cmp_ne_u32_e32 vcc_lo, 1, v4
	s_mov_b32 s2, -1
	s_cbranch_vccnz .LBB3_404
; %bb.401:
	v_add_nc_u32_e32 v4, s1, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s10, v4
	s_and_b32 s1, s0, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_403
; %bb.402:
	v_ashrrev_i32_e32 v6, 31, v4
	v_mul_lo_u32 v7, s13, v4
	s_wait_dscnt 0x1
	v_mad_co_u64_u32 v[4:5], null, s12, v4, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v6, s12, v6
	v_add3_u32 v5, v5, v6, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, v1, v5, vcc_lo
	s_wait_dscnt 0x0
	global_store_b32 v[0:1], v2, off
.LBB3_403:                              ; %Flow
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s2, 0
.LBB3_404:                              ; %Flow860
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_406
; %bb.405:
	s_mul_i32 s0, s8, 0x1f8
	s_wait_dscnt 0x0
	buffer_store_b32 v2, v3, s[4:7], s0 offen
.LBB3_406:
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
.LBB3_407:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb0EEvPKcPK12block_i4_128Pfiii.exit
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
		.amdhsa_next_free_vgpr 179
		.amdhsa_next_free_sgpr 32
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_vgpr, 179
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.numbered_sgpr, 32
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 17364
; TotalNumSgprs: 34
; NumVgprs: 179
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 179
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
	.type	__hip_cuid_5bd98453e0709f10,@object ; @__hip_cuid_5bd98453e0709f10
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5bd98453e0709f10
__hip_cuid_5bd98453e0709f10:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5bd98453e0709f10, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5bd98453e0709f10
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
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     179
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
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     179
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
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     179
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
