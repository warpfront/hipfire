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
	v_or_b32_e32 v7, 3, v6
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_mov_b32 s8, ttmp7
	s_ashr_i32 s9, ttmp7, 31
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s2, v7
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
	v_mov_b32_e32 v13, 0x7149f2ca
	v_div_fixup_f32 v14, v6, 0x40e00000, v12
	v_mov_b32_e32 v6, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v14, 0.5, v14
.LBB0_5:                                ; %.preheader.preheader.i
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cvt_f32_u32 s0, s5
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 8
	v_div_scale_f32 v15, null, s9, s9, s0
	v_div_scale_f32 v18, vcc_lo, s0, 0x40e00000, s0
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v16, v15
	v_xor_b32_e32 v15, 0x80000000, v15
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v17, v15, v16, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v17, v16
	v_mul_f32_e32 v17, v18, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v19, v15, v17, v18
	v_fmac_f32_e32 v17, v19, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v18, v15, v17
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v18, v16, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v15, v15, 0x40e00000, s0
	v_add_f32_e32 v15, 1.0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v14, v15
	v_div_scale_f32 v16, null, v15, v15, v1
	v_div_scale_f32 v17, null, v15, v15, v2
	v_div_scale_f32 v18, null, v15, v15, v3
	v_div_scale_f32 v19, null, v15, v15, v4
	v_div_scale_f32 v25, s0, v2, v15, v2
	v_rcp_f32_e32 v20, v16
	v_rcp_f32_e32 v21, v17
	v_rcp_f32_e32 v22, v18
	v_rcp_f32_e32 v23, v19
	v_div_scale_f32 v24, vcc_lo, v1, v15, v1
	v_div_scale_f32 v29, s1, v3, v15, v3
	v_fma_f32 v26, -v16, v20, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v27, -v17, v21, 1.0
	v_fma_f32 v28, -v18, v22, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v30, -v19, v23, 1.0
	v_dual_fmac_f32 v20, v26, v20 :: v_dual_fmac_f32 v21, v27, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v22, v28, v22 :: v_dual_fmac_f32 v23, v30, v23
	v_div_scale_f32 v26, s2, v4, v15, v4
	v_mul_f32_e32 v28, v25, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v30, v29, v22
	v_fma_f32 v33, -v17, v28, v25
	v_mul_f32_e32 v27, v24, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v34, -v18, v30, v29
	v_fmac_f32_e32 v28, v33, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v32, -v16, v27, v24
	v_fmac_f32_e32 v30, v34, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v17, -v17, v28, v25
	v_fmac_f32_e32 v27, v32, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v30, v29
	v_fma_f32 v16, -v16, v27, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v16, v16, v20, v27
	s_mov_b32 vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v17, v17, v21, v28
	s_mov_b32 vcc_lo, s1
	v_div_fixup_f32 v16, v16, v15, v1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v22, v30
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v17, v17, v15, v2
	v_rndne_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v18, v18, v15, v3
	v_rndne_f32_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v16, v16, s10, 0x40e00000
	v_rndne_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v17, v17, s10, 0x40e00000
	v_fma_f32 v16, -v16, v15, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v18, v18, s10, 0x40e00000
	v_fma_f32 v17, -v17, v15, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v16, v16, v16, 0
	v_mul_f32_e32 v31, v26, v23
	v_fma_f32 v18, -v18, v15, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v16, v17, v17
	v_fma_f32 v35, -v19, v31, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v16, v18, v18 :: v_dual_fmac_f32 v31, v35, v23
	v_fma_f32 v19, -v19, v31, v26
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v19, v19, v23, v31
	v_div_fixup_f32 v19, v19, v15, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v19, v19
	v_med3_num_f32 v19, v19, s10, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, -v19, v15, v4
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
	v_cmp_lt_f32_e32 vcc_lo, v16, v13
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v13, v13, v16 :: v_dual_cndmask_b32 v6, v6, v15
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
	v_div_scale_f32 v16, vcc_lo, v1, v6, v1
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v13
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v13, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v16, v14
	v_fma_f32 v17, -v13, v15, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v17, v14
	v_fma_f32 v13, -v13, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v13, v13, v14, v15
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
	v_div_scale_f32 v15, vcc_lo, v2, v6, v2
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v12, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v12, 1.0
	v_fmac_f32_e32 v12, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v12
	v_fma_f32 v16, -v1, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v12
	v_fma_f32 v1, -v1, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v12, v14
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
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, 0
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_div_scale_f32 v2, null, v6, v6, v3
	v_div_scale_f32 v16, vcc_lo, v3, v6, v3
	s_mov_b32 s2, 0xc1000000
	v_rcp_f32_e32 v14, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v2, v14, 1.0
	v_fmac_f32_e32 v14, v15, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v15, v16, v14
	v_fma_f32 v17, -v2, v15, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v15, v17, v14
	v_fma_f32 v2, -v2, v15, v16
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v2, v2, v14, v15
	v_div_fixup_f32 v2, v2, v6, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s2, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v2, v2
.LBB0_12:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_div_scale_f32 v1, null, v6, v6, v4
	v_div_scale_f32 v15, vcc_lo, v4, v6, v4
	s_mov_b32 s0, 0xc1000000
	v_rcp_f32_e32 v3, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v1, v3, 1.0
	v_fmac_f32_e32 v3, v14, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v14, v15, v3
	v_fma_f32 v16, -v1, v14, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v14, v16, v3
	v_fma_f32 v1, -v1, v14, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v1, v1, v3, v14
	v_div_fixup_f32 v1, v1, v6, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s0, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v1, v1
.LBB0_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v3, v12, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add3_u32 v3, v3, v2, v1
	ds_bpermute_b32 v4, v7, v3
	v_ashrrev_i32_e32 v7, 31, v5
	v_lshrrev_b32_e32 v7, 27, v7
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v8, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v3, v3, v4
	ds_bpermute_b32 v4, v10, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v8, v3, v4
	v_add_nc_u32_e32 v3, v5, v7
	ds_bpermute_b32 v5, v11, v8
	v_ashrrev_i32_e32 v3, 5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[3:4], null, v3, s3, 0
	v_mad_co_u64_u32 v[14:15], null, 0x48, v3, s[6:7]
	v_and_b32_e32 v3, 15, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v1, v1, 4, v3
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v8, v5
	v_mad_co_u64_u32 v[15:16], null, 0x48, v4, v[15:16]
	v_and_b32_e32 v5, 15, v13
	v_lshlrev_b16 v4.l, 8, v1.l
	ds_bpermute_b32 v3, v9, v2
	v_and_b32_e32 v9, 31, v0
	v_lshl_or_b32 v5, v12, 4, v5
	v_mad_co_i64_i32 v[0:1], null, 0x48, s8, v[14:15]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b16 v4.h, 0xff, v5.l
	v_lshlrev_b32_e32 v5, 1, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b16 v4.l, v4.h, v4.l
	v_add_co_u32 v7, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v1, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	global_store_b16 v[7:8], v4, off offset:8
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v2, v3
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
; codeLenInByte = 2304
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
	v_mov_b32_e32 v17, v0
	s_lshl_b32 s11, ttmp7, 7
	s_delay_alu instid0(VALU_DEP_1)
	v_lshrrev_b32_e32 v18, 5, v17
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s22, s12
	s_cselect_b32 s0, -1, 0
	s_cmp_lt_i32 s11, s14
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
; %bb.2:                                ; %.preheader476.i
	s_ashr_i32 s0, s13, 31
	s_add_co_i32 s2, s12, -1
	s_lshr_b32 s0, s0, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s0, s13, s0
	s_ashr_i32 s10, s0, 8
	s_mov_b32 s0, exec_lo
	s_mul_i32 s4, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v17
	s_cbranch_execz .LBB1_6
; %bb.3:                                ; %.lr.ph.i
	v_lshrrev_b32_e32 v1, 1, v17
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v3, 1, v17
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v5, s11, v1
	v_lshlrev_b32_e32 v2, 2, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v4, v2
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_5:                                ; %.preheader471.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s22, v1
	v_lshlrev_b32_e32 v3, 4, v3
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	v_add3_u32 v1, 0, v1, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[5:6], null, s4, v5, s[16:17]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v3, v3, v5
	v_cvt_f32_f16_e32 v3, v3.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v2, 0, v3, vcc_lo
	ds_store_2addr_stride64_b32 v1, v4, v2 offset0:48 offset1:56
.LBB1_6:                                ; %Flow1906
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v9, 2, v17
	v_and_b32_e32 v1, 3, v17
	s_add_co_i32 s3, s14, -1
	v_lshlrev_b32_e32 v13, 4, v17
	v_bfe_u32 v12, v17, 1, 1
	v_add_nc_u32_e32 v10, s11, v9
	v_add_nc_u32_e32 v2, s22, v9
	v_lshlrev_b32_e32 v1, 3, v1
	v_and_b32_e32 v13, 16, v13
	v_or_b32_e32 v14, 8, v18
	v_add_nc_u32_e32 v11, 64, v10
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v3, s3, v10
	v_add_nc_u32_e32 v4, 64, v2
	v_min_i32_e32 v2, s2, v2
	v_and_or_b32 v9, v9, 15, v13
	v_min_i32_e32 v5, s3, v11
	v_and_or_b32 v13, v18, 6, v12
	v_and_or_b32 v12, v14, 14, v12
	v_mad_co_u64_u32 v[127:128], null, 0x48, v3, v[1:2]
	v_min_i32_e32 v3, s2, v4
	v_mad_co_u64_u32 v[128:129], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[129:130], null, s4, v2, v[1:2]
	v_lshlrev_b32_e32 v9, 3, v9
	v_cmp_gt_i32_e64 s0, s14, v10
	v_mad_co_u64_u32 v[130:131], null, s4, v3, v[1:2]
	v_cmp_gt_i32_e64 s1, s14, v11
	v_and_b32_e32 v0, 15, v17
	s_clause 0x1
	global_load_b64 v[1:2], v127, s[18:19] offset:8
	global_load_b64 v[3:4], v128, s[18:19] offset:8
	s_clause 0x1
	global_load_b64 v[5:6], v129, s[16:17] offset:8
	global_load_b64 v[7:8], v130, s[16:17] offset:8
	v_lshl_or_b32 v13, v13, 8, v9
	v_lshl_or_b32 v9, v12, 8, v9
	v_bfe_u32 v15, v17, 4, 1
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_mov_b32_e32 v25, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v160, 0
	v_mov_b32_e32 v159, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v28, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v148, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v152, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v183, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v185, 0
	v_add_nc_u32_e32 v10, 0, v13
	v_dual_mov_b32 v186, 0 :: v_dual_add_nc_u32 v9, 0, v9
	v_mov_b32_e32 v179, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v10, off offset:52
	scratch_store_b32 off, v9, off offset:60
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v4, 0, v4, s1
	v_cndmask_b32_e64 v3, 0, v3, s1
	s_wait_loadcnt 0x1
	ds_store_2addr_stride64_b64 v10, v[1:2], v[5:6] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v9, v[3:4], v[7:8] offset1:8
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_18
; %bb.7:                                ; %.preheader470.lr.ph.i
	scratch_store_b32 off, v0, off offset:124 ; 4-byte Folded Spill
	v_dual_mov_b32 v0, 0 :: v_dual_lshlrev_b32 v7, 3, v0
	v_and_b32_e32 v5, 1, v17
	v_lshrrev_b32_e32 v4, 1, v17
	v_and_b32_e32 v1, 31, v17
	v_lshrrev_b32_e32 v2, 6, v17
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, 0
	v_add_nc_u32_e32 v8, s22, v4
	v_add_nc_u32_e32 v9, s11, v4
	v_lshlrev_b32_e32 v4, 3, v4
	v_lshlrev_b32_e32 v10, 2, v5
	scratch_store_b32 off, v0, off offset:68 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 4, v5
	v_min_i32_e32 v11, s2, v8
	v_lshlrev_b32_e32 v1, 3, v1
	scratch_store_b32 off, v15, off offset:120 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v6, 6, v15
	scratch_store_b32 off, v0, off offset:84 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 2, v5
	v_mad_co_u64_u32 v[14:15], null, s4, v11, s[16:17]
	v_bfe_u32 v3, v17, 5, 1
	v_lshlrev_b32_e32 v13, 8, v2
	v_lshl_or_b32 v68, v2, 10, v1
	scratch_store_b32 off, v0, off offset:100 ; 4-byte Folded Spill
	v_min_i32_e32 v0, s3, v9
	v_ashrrev_i32_e32 v2, 31, v11
	v_lshl_add_u32 v12, v3, 11, 0
	v_lshlrev_b32_e32 v3, 9, v3
	v_mov_b32_e32 v179, 0
	scratch_store_b32 off, v0, off offset:104 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v4, v10
	v_mad_co_u64_u32 v[15:16], null, s4, v2, v[15:16]
	v_dual_mov_b32 v186, 0 :: v_dual_mov_b32 v185, 0
	v_mov_b32_e32 v184, 0
	scratch_store_b32 off, v0, off offset:108 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v13, v6
	v_dual_mov_b32 v182, 0 :: v_dual_mov_b32 v183, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v134, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v132, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v154, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v25, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_cmp_gt_i32_e64 s2, s14, v9
	v_cmp_gt_i32_e64 s3, s12, v8
	scratch_store_b32 off, v0, off offset:76 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v3, v7
	v_add_nc_u32_e32 v197, v12, v1
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s30, 0x1000
	s_movk_i32 s25, 0x3000
	s_movk_i32 s26, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_clause 0x3                            ; 24-byte Folded Spill
	scratch_store_b32 off, v18, off offset:116
	scratch_store_b32 off, v17, off offset:112
	scratch_store_b96 off, v[14:16], off offset:88
	scratch_store_b32 off, v0, off offset:80
	s_branch .LBB1_9
.LBB1_8:                                ;   in Loop: Header=BB1_9 Depth=1
	s_and_b32 vcc_lo, exec_lo, s7
	s_mov_b32 s6, s27
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_17
.LBB1_9:                                ; %.preheader470.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_11 Depth 2
	s_add_co_i32 s27, s6, 1
	s_mov_b32 s7, s5
	s_lshl_b32 s28, s6, 1
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_cmp_eq_u32 s27, s10
	s_mov_b32 s33, 0
	s_mov_b32 s29, -1
	s_mov_b32 s31, 0
	s_cselect_b32 s7, -1, 0
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_branch .LBB1_11
.LBB1_10:                               ;   in Loop: Header=BB1_11 Depth=2
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v24, v24
	v_mul_f32_e32 v238, v94, v90
	v_dual_mul_f32 v239, v94, v86 :: v_dual_mul_f32 v242, v94, v84
	v_dual_mul_f32 v243, v94, v82 :: v_dual_mul_f32 v246, v94, v80
	v_dual_mul_f32 v169, v92, v84 :: v_dual_mul_f32 v178, v92, v82
	v_dual_mul_f32 v191, v92, v80 :: v_dual_mul_f32 v194, v92, v78
	v_dual_mul_f32 v65, v92, v76 :: v_dual_mul_f32 v66, v92, v74
	v_dual_mul_f32 v205, v110, v92 :: v_dual_mul_f32 v206, v108, v92
	v_cvt_f32_i32_e32 v207, v49
	v_cvt_f32_i32_e32 v208, v50
	v_dual_mul_f32 v49, v111, v92 :: v_dual_mul_f32 v50, v109, v92
	v_dual_mul_f32 v209, v104, v92 :: v_dual_mul_f32 v210, v106, v92
	v_cvt_f32_i32_e32 v211, v51
	v_cvt_f32_i32_e32 v212, v52
	v_dual_mul_f32 v51, v105, v92 :: v_dual_mul_f32 v52, v107, v92
	v_mul_f32_e32 v213, v102, v92
	v_mul_f32_e32 v214, v98, v92
	v_cvt_f32_i32_e32 v215, v53
	v_cvt_f32_i32_e32 v216, v54
	v_mul_f32_e32 v53, v103, v92
	v_dual_mul_f32 v54, v99, v92 :: v_dual_mul_f32 v217, v100, v92
	v_cvt_f32_i32_e32 v218, v55
	v_dual_mul_f32 v55, v101, v92 :: v_dual_mul_f32 v234, v96, v92
	v_mul_f32_e32 v235, v97, v92
	v_cvt_f32_i32_e32 v95, v95
	v_mul_f32_e32 v240, v94, v91
	v_dual_mul_f32 v241, v94, v87 :: v_dual_mul_f32 v244, v94, v85
	v_dual_mul_f32 v245, v94, v83 :: v_dual_mul_f32 v248, v94, v81
	v_cvt_f32_i32_e32 v93, v93
	v_mul_f32_e32 v253, v92, v90
	v_dual_mul_f32 v254, v92, v86 :: v_dual_mul_f32 v255, v92, v91
	v_dual_mul_f32 v120, v92, v87 :: v_dual_mul_f32 v189, v92, v85
	v_mul_f32_e32 v0, v92, v83
	v_mul_f32_e32 v190, v92, v81
	v_dual_mul_f32 v192, v92, v79 :: v_dual_mul_f32 v195, v92, v77
	v_dual_mul_f32 v92, v92, v75 :: v_dual_mul_f32 v25, v238, v25
	v_dual_mul_f32 v26, v239, v26 :: v_dual_mul_f32 v27, v242, v27
	v_dual_mul_f32 v28, v243, v28 :: v_dual_mul_f32 v29, v246, v29
	v_dual_mul_f32 v19, v169, v19 :: v_dual_mul_f32 v20, v178, v20
	v_dual_mul_f32 v21, v191, v21 :: v_dual_mul_f32 v22, v194, v22
	v_dual_mul_f32 v23, v65, v23 :: v_dual_mul_f32 v24, v66, v24
	v_dual_fmac_f32 v25, v240, v95 :: v_dual_fmac_f32 v26, v241, v95
	v_dual_fmac_f32 v27, v244, v95 :: v_dual_fmac_f32 v28, v245, v95
	v_fmac_f32_e32 v29, v248, v95
	v_dual_fmac_f32 v19, v189, v93 :: v_dual_fmac_f32 v20, v0, v93
	v_dual_fmac_f32 v21, v190, v93 :: v_dual_fmac_f32 v22, v192, v93
	v_dual_fmac_f32 v23, v195, v93 :: v_dual_fmac_f32 v24, v92, v93
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_dual_add_f32 v152, v152, v25 :: v_dual_add_f32 v151, v151, v26
	v_dual_add_f32 v149, v149, v27 :: v_dual_add_f32 v150, v150, v28
	v_dual_add_f32 v148, v148, v29 :: v_dual_add_f32 v141, v141, v19
	v_dual_add_f32 v142, v142, v20 :: v_dual_add_f32 v139, v139, v21
	v_dual_add_f32 v140, v140, v22 :: v_dual_add_f32 v137, v137, v23
	v_add_f32_e32 v138, v138, v24
	s_clause 0xa                            ; 44-byte Folded Reload
	scratch_load_b32 v28, off, off offset:36 th:TH_LOAD_LU
	scratch_load_b32 v26, off, off offset:28 th:TH_LOAD_LU
	scratch_load_b32 v27, off, off offset:32 th:TH_LOAD_LU
	scratch_load_b32 v29, off, off offset:44 th:TH_LOAD_LU
	scratch_load_b32 v25, off, off offset:24 th:TH_LOAD_LU
	scratch_load_b32 v23, off, off offset:16 th:TH_LOAD_LU
	scratch_load_b32 v24, off, off offset:20 th:TH_LOAD_LU
	scratch_load_b32 v21, off, off offset:8 th:TH_LOAD_LU
	scratch_load_b32 v22, off, off offset:12 th:TH_LOAD_LU
	scratch_load_b32 v20, off, off offset:4 th:TH_LOAD_LU
	scratch_load_b32 v19, off, off th:TH_LOAD_LU
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v114, v57
	v_cvt_f32_i32_e32 v115, v58
	v_dual_mul_f32 v116, v104, v94 :: v_dual_mul_f32 v117, v106, v94
	v_cvt_f32_i32_e32 v118, v59
	v_cvt_f32_i32_e32 v119, v60
	v_mul_f32_e32 v198, v102, v94
	v_mul_f32_e32 v199, v98, v94
	v_cvt_f32_i32_e32 v200, v61
	v_cvt_f32_i32_e32 v201, v62
	v_mul_f32_e32 v202, v100, v94
	v_mul_f32_e32 v203, v96, v94
	v_cvt_f32_i32_e32 v204, v63
	v_cvt_f32_i32_e32 v64, v64
	v_cvt_f32_i32_e32 v56, v56
	v_dual_mul_f32 v219, v110, v88 :: v_dual_mul_f32 v220, v108, v88
	v_cvt_f32_i32_e32 v221, v41
	v_cvt_f32_i32_e32 v222, v42
	v_dual_mul_f32 v223, v104, v88 :: v_dual_mul_f32 v224, v106, v88
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v15, v15
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v227, v102, v88
	v_dual_mul_f32 v228, v98, v88 :: v_dual_mul_f32 v231, v100, v88
	v_mul_f32_e32 v110, v110, v72
	v_mul_f32_e32 v108, v108, v72
	v_mul_f32_e32 v104, v104, v72
	v_mul_f32_e32 v106, v106, v72
	v_mul_f32_e32 v102, v102, v72
	v_mul_f32_e32 v98, v98, v72
	v_mul_f32_e32 v100, v100, v72
	v_mul_f32_e32 v236, v96, v88
	v_dual_mul_f32 v96, v96, v72 :: v_dual_mul_f32 v247, v94, v78
	v_dual_mul_f32 v250, v94, v76 :: v_dual_mul_f32 v251, v94, v74
	v_mul_f32_e32 v67, v88, v90
	v_mul_f32_e32 v196, v88, v86
	v_dual_mul_f32 v70, v88, v84 :: v_dual_mul_f32 v71, v88, v82
	v_mul_f32_e32 v193, v88, v80
	v_dual_mul_f32 v121, v88, v78 :: v_dual_mul_f32 v124, v88, v76
	v_mul_f32_e32 v125, v88, v74
	v_mul_f32_e32 v90, v72, v90
	v_mul_f32_e32 v86, v72, v86
	v_mul_f32_e32 v84, v72, v84
	v_mul_f32_e32 v82, v72, v82
	v_mul_f32_e32 v80, v72, v80
	v_mul_f32_e32 v78, v72, v78
	v_mul_f32_e32 v76, v72, v76
	v_mul_f32_e32 v74, v72, v74
	v_dual_mul_f32 v57, v111, v94 :: v_dual_mul_f32 v58, v109, v94
	v_dual_mul_f32 v59, v105, v94 :: v_dual_mul_f32 v60, v107, v94
	v_mul_f32_e32 v61, v103, v94
	v_dual_mul_f32 v62, v99, v94 :: v_dual_mul_f32 v63, v101, v94
	v_dual_mul_f32 v41, v111, v88 :: v_dual_mul_f32 v42, v109, v88
	v_dual_mul_f32 v225, v105, v88 :: v_dual_mul_f32 v226, v107, v88
	v_mul_f32_e32 v229, v103, v88
	v_mul_f32_e32 v230, v99, v88
	v_mul_f32_e32 v232, v101, v88
	v_mul_f32_e32 v111, v111, v72
	v_mul_f32_e32 v109, v109, v72
	v_mul_f32_e32 v105, v105, v72
	v_mul_f32_e32 v107, v107, v72
	v_mul_f32_e32 v103, v103, v72
	v_mul_f32_e32 v99, v99, v72
	v_mul_f32_e32 v101, v101, v72
	v_mul_f32_e32 v233, v97, v94
	v_mul_f32_e32 v237, v97, v88
	v_mul_f32_e32 v97, v97, v72
	v_dual_mul_f32 v249, v94, v79 :: v_dual_mul_f32 v252, v94, v77
	v_mul_f32_e32 v94, v94, v75
	v_cvt_f32_i32_e32 v89, v89
	v_mul_f32_e32 v68, v88, v91
	v_mul_f32_e32 v69, v88, v87
	v_dual_mul_f32 v187, v88, v85 :: v_dual_mul_f32 v188, v88, v83
	v_dual_mul_f32 v122, v88, v81 :: v_dual_mul_f32 v123, v88, v79
	v_mul_f32_e32 v126, v88, v77
	v_mul_f32_e32 v88, v88, v75
	v_mul_f32_e32 v91, v72, v91
	v_mul_f32_e32 v87, v72, v87
	v_mul_f32_e32 v85, v72, v85
	v_mul_f32_e32 v83, v72, v83
	v_mul_f32_e32 v81, v72, v81
	v_mul_f32_e32 v79, v72, v79
	v_dual_mul_f32 v77, v72, v77 :: v_dual_mul_f32 v72, v72, v75
	v_cvt_f32_i32_e32 v73, v73
	v_dual_mul_f32 v75, v112, v114 :: v_dual_mul_f32 v112, v113, v115
	v_dual_mul_f32 v113, v116, v118 :: v_dual_mul_f32 v114, v117, v119
	v_dual_mul_f32 v115, v198, v200 :: v_dual_mul_f32 v116, v199, v201
	v_mul_f32_e32 v117, v202, v204
	v_mul_f32_e32 v64, v203, v64
	v_dual_mul_f32 v118, v205, v207 :: v_dual_mul_f32 v119, v206, v208
	v_dual_mul_f32 v198, v209, v211 :: v_dual_mul_f32 v199, v210, v212
	v_dual_mul_f32 v200, v213, v215 :: v_dual_mul_f32 v201, v214, v216
	v_mul_f32_e32 v202, v217, v218
	v_dual_mul_f32 v56, v234, v56 :: v_dual_mul_f32 v203, v219, v221
	v_dual_mul_f32 v204, v220, v222 :: v_dual_mul_f32 v43, v223, v43
	v_dual_mul_f32 v44, v224, v44 :: v_dual_mul_f32 v45, v227, v45
	v_dual_mul_f32 v46, v228, v46 :: v_dual_mul_f32 v47, v231, v47
	v_dual_mul_f32 v48, v236, v48 :: v_dual_mul_f32 v33, v110, v33
	v_mul_f32_e32 v34, v108, v34
	v_dual_mul_f32 v35, v104, v35 :: v_dual_mul_f32 v36, v106, v36
	v_mul_f32_e32 v37, v102, v37
	v_dual_mul_f32 v38, v98, v38 :: v_dual_mul_f32 v39, v100, v39
	v_mul_f32_e32 v40, v96, v40
	v_dual_mul_f32 v30, v247, v30 :: v_dual_mul_f32 v31, v250, v31
	v_dual_mul_f32 v32, v251, v32 :: v_dual_mul_f32 v17, v253, v17
	v_dual_mul_f32 v18, v254, v18 :: v_dual_mul_f32 v9, v67, v9
	v_dual_mul_f32 v10, v196, v10 :: v_dual_mul_f32 v11, v70, v11
	v_dual_mul_f32 v12, v71, v12 :: v_dual_mul_f32 v13, v193, v13
	v_dual_mul_f32 v14, v121, v14 :: v_dual_mul_f32 v15, v124, v15
	v_dual_mul_f32 v16, v125, v16 :: v_dual_mul_f32 v1, v90, v1
	v_dual_mul_f32 v2, v86, v2 :: v_dual_mul_f32 v3, v84, v3
	v_dual_mul_f32 v4, v82, v4 :: v_dual_mul_f32 v5, v80, v5
	v_dual_mul_f32 v6, v78, v6 :: v_dual_mul_f32 v7, v76, v7
	v_dual_mul_f32 v8, v74, v8 :: v_dual_fmac_f32 v75, v57, v95
	v_dual_fmac_f32 v112, v58, v95 :: v_dual_fmac_f32 v113, v59, v95
	v_dual_fmac_f32 v114, v60, v95 :: v_dual_fmac_f32 v115, v61, v95
	v_dual_fmac_f32 v116, v62, v95 :: v_dual_fmac_f32 v117, v63, v95
	v_fmac_f32_e32 v64, v233, v95
	v_dual_fmac_f32 v118, v49, v93 :: v_dual_fmac_f32 v119, v50, v93
	v_dual_fmac_f32 v198, v51, v93 :: v_dual_fmac_f32 v199, v52, v93
	v_dual_fmac_f32 v200, v53, v93 :: v_dual_fmac_f32 v201, v54, v93
	v_fmac_f32_e32 v202, v55, v93
	v_fmac_f32_e32 v56, v235, v93
	v_dual_fmac_f32 v203, v41, v89 :: v_dual_fmac_f32 v204, v42, v89
	v_dual_fmac_f32 v43, v225, v89 :: v_dual_fmac_f32 v44, v226, v89
	v_dual_fmac_f32 v45, v229, v89 :: v_dual_fmac_f32 v46, v230, v89
	v_dual_fmac_f32 v47, v232, v89 :: v_dual_fmac_f32 v48, v237, v89
	v_dual_fmac_f32 v33, v111, v73 :: v_dual_fmac_f32 v34, v109, v73
	v_dual_fmac_f32 v35, v105, v73 :: v_dual_fmac_f32 v36, v107, v73
	v_fmac_f32_e32 v37, v103, v73
	v_dual_fmac_f32 v38, v99, v73 :: v_dual_fmac_f32 v39, v101, v73
	v_fmac_f32_e32 v40, v97, v73
	v_dual_fmac_f32 v30, v249, v95 :: v_dual_fmac_f32 v31, v252, v95
	v_dual_fmac_f32 v32, v94, v95 :: v_dual_fmac_f32 v17, v255, v93
	v_fmac_f32_e32 v18, v120, v93
	v_dual_fmac_f32 v9, v68, v89 :: v_dual_fmac_f32 v10, v69, v89
	v_dual_fmac_f32 v11, v187, v89 :: v_dual_fmac_f32 v12, v188, v89
	v_dual_fmac_f32 v13, v122, v89 :: v_dual_fmac_f32 v14, v123, v89
	v_dual_fmac_f32 v15, v126, v89 :: v_dual_fmac_f32 v16, v88, v89
	v_fmac_f32_e32 v1, v91, v73
	v_dual_fmac_f32 v2, v87, v73 :: v_dual_fmac_f32 v3, v85, v73
	v_dual_fmac_f32 v4, v83, v73 :: v_dual_fmac_f32 v5, v81, v73
	v_dual_fmac_f32 v6, v79, v73 :: v_dual_fmac_f32 v7, v77, v73
	v_dual_fmac_f32 v8, v72, v73 :: v_dual_add_f32 v179, v179, v75
	v_dual_add_f32 v186, v186, v112 :: v_dual_add_f32 v185, v185, v113
	v_add_f32_e32 v184, v184, v114
	v_dual_add_f32 v182, v182, v115 :: v_dual_add_f32 v183, v183, v116
	v_dual_add_f32 v180, v180, v117 :: v_dual_add_f32 v181, v181, v64
	v_dual_add_f32 v176, v176, v118 :: v_dual_add_f32 v177, v177, v119
	v_dual_add_f32 v174, v174, v198 :: v_dual_add_f32 v175, v175, v199
	v_dual_add_f32 v172, v172, v200 :: v_dual_add_f32 v173, v173, v201
	v_dual_add_f32 v170, v170, v202 :: v_dual_add_f32 v171, v171, v56
	v_dual_add_f32 v167, v167, v203 :: v_dual_add_f32 v168, v168, v204
	v_dual_add_f32 v165, v165, v43 :: v_dual_add_f32 v166, v166, v44
	v_dual_add_f32 v163, v163, v45 :: v_dual_add_f32 v164, v164, v46
	v_dual_add_f32 v161, v161, v47 :: v_dual_add_f32 v162, v162, v48
	v_dual_add_f32 v159, v159, v33 :: v_dual_add_f32 v160, v160, v34
	v_dual_add_f32 v157, v157, v35 :: v_dual_add_f32 v158, v158, v36
	v_dual_add_f32 v155, v155, v37 :: v_dual_add_f32 v156, v156, v38
	v_dual_add_f32 v153, v153, v39 :: v_dual_add_f32 v154, v154, v40
	v_add_f32_e32 v147, v147, v30
	v_dual_add_f32 v145, v145, v31 :: v_dual_add_f32 v146, v146, v32
	v_dual_add_f32 v143, v143, v17 :: v_dual_add_f32 v144, v144, v18
	v_dual_add_f32 v135, v135, v9 :: v_dual_add_f32 v136, v136, v10
	v_dual_add_f32 v133, v133, v11 :: v_dual_add_f32 v134, v134, v12
	v_add_f32_e32 v132, v132, v14
	v_mov_b32_e32 v68, v131
	s_xor_b32 s4, s29, -1
	s_mov_b32 s33, 1
	s_mov_b32 s29, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s31, -1
	s_wait_loadcnt 0xa
	v_add_f32_e32 v28, v28, v13
	s_wait_loadcnt 0x8
	v_dual_add_f32 v26, v26, v15 :: v_dual_add_f32 v27, v27, v16
	s_wait_loadcnt 0x7
	v_add_f32_e32 v29, v29, v1
	s_wait_loadcnt 0x6
	v_add_f32_e32 v25, v25, v2
	s_wait_loadcnt 0x4
	v_dual_add_f32 v23, v23, v3 :: v_dual_add_f32 v24, v24, v4
	s_wait_loadcnt 0x2
	v_dual_add_f32 v21, v21, v5 :: v_dual_add_f32 v22, v22, v6
	s_wait_loadcnt 0x0
	v_dual_add_f32 v20, v20, v7 :: v_dual_add_f32 v19, v19, v8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_8
.LBB1_11:                               ;   Parent Loop BB1_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s4, s33, s28
	s_lshl_b32 s34, s33, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_mov_b32 s35, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_add_nc_u64 s[34:35], s[8:9], s[34:35]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	v_add_co_u32 v1, s38, s34, v129
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s39, s36, v127
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s37, 0, s39
	v_add_co_ci_u32_e64 v2, null, s35, 0, s38
	v_add_co_u32 v5, s36, s36, v128
	v_add_co_u32 v7, s34, s34, v130
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s37, 0, s36
	global_load_b64 v[96:97], v[3:4], off offset:40
	v_add_co_ci_u32_e64 v8, null, s35, 0, s34
	global_load_b64 v[112:113], v[1:2], off offset:40
	v_add_nc_u32_e32 v1, s30, v68
	s_clause 0xa                            ; 44-byte Folded Spill
	scratch_store_b32 off, v29, off offset:44
	scratch_store_b32 off, v28, off offset:36
	scratch_store_b32 off, v27, off offset:32
	scratch_store_b32 off, v26, off offset:28
	scratch_store_b32 off, v25, off offset:24
	scratch_store_b32 off, v24, off offset:20
	scratch_store_b32 off, v23, off offset:16
	scratch_store_b32 off, v22, off offset:12
	scratch_store_b32 off, v21, off offset:8
	scratch_store_b32 off, v20, off offset:4
	scratch_store_b32 off, v19, off
	global_load_b64 v[98:99], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_b64 v[72:75], v197 offset1:32
	ds_load_2addr_b64 v[76:79], v1 offset1:32
	ds_load_2addr_b64 v[80:83], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[84:87], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[88:91], v197 offset0:192 offset1:224
	ds_load_2addr_b64 v[92:95], v1 offset0:64 offset1:96
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off offset:52 ; 4-byte Folded Reload
	s_and_b32 s30, s31, s7
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s30
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[92:93], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[92:93], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[92:93], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[92:93], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[78:79], v[74:75], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[78:79], v[82:83], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[78:79], v[86:87], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[78:79], v[90:91], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[94:95], v[74:75], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[94:95], v[82:83], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[94:95], v[86:87], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[94:95], v[90:91], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_cndmask_b32_e64 v119, 0, v97, s0
	v_cndmask_b32_e64 v118, 0, v96, s0
	v_cndmask_b32_e64 v117, 0, v99, s1
	v_cndmask_b32_e64 v116, 0, v98, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:16
	scratch_load_b32 v0, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_11 Depth=2
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b32 v0, off, off offset:104
	scratch_load_b96 v[65:67], off, off offset:88
	s_add_co_i32 s4, s4, 1
	s_xor_b32 s38, s33, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], s[14:15]
	s_add_co_i32 s4, s33, s6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[36:37], 0x48
	s_lshl_b32 s34, s38, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[36:37], s[18:19], s[36:37]
	s_lshl_b32 s33, s38, 2
	s_mul_u64 s[38:39], s[4:5], 0x88
	s_mulk_i32 s4, 0x88
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s40, s36, v128
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s37, 0, s40
	s_mov_b32 s35, s5
	s_add_nc_u64 s[38:39], s[16:17], s[38:39]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[38:39], s[34:35]
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[76:77], null, 0x48, v0, s[36:37]
	scratch_load_b32 v0, off, off offset:100 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_add_co_u32 v78, vcc_lo, v65, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v66, vcc_lo
	v_add_co_u32 v72, s4, s36, v127
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v78, vcc_lo, v78, s33
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s37, 0, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v79, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, s4, s34, v129
	global_load_b64 v[118:119], v[72:73], off offset:8
	v_add_co_u32 v82, s33, s34, v130
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s35, 0, s4
	v_add_co_ci_u32_e64 v83, null, s35, 0, s33
	s_wait_loadcnt 0x1
	v_add_co_u32 v72, vcc_lo, v76, v0
	global_load_b64 v[116:117], v[74:75], off offset:8
	global_load_b32 v0, v[78:79], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, 0, v77, vcc_lo
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:68 ; 4-byte Folded Spill
	s_clause 0x1
	global_load_b64 v[112:113], v[80:81], off offset:8
	global_load_b64 v[114:115], v[82:83], off offset:8
	global_load_b32 v0, v[72:73], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
.LBB1_13:                               ; %.preheader468.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	scratch_load_b32 v0, off, off offset:76 ; 4-byte Folded Reload
	s_xor_b32 s4, s30, -1
	s_and_b32 s30, s29, exec_lo
	s_cselect_b32 s30, s26, 0x3c00
	v_add_nc_u32_e32 v72, 0, v68
	ds_load_2addr_b64 v[198:201], v197 offset1:32
	ds_load_2addr_b64 v[202:205], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[206:209], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[210:213], v197 offset0:192 offset1:224
	v_add_nc_u32_e32 v72, 0x2000, v72
	s_cselect_b32 s33, s25, 0x3400
	v_mov_b32_e32 v131, v68
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v74, s30, v0
	scratch_load_b32 v0, off, off offset:80 ; 4-byte Folded Reload
	s_movk_i32 s30, 0x2000
	ds_load_2addr_b32 v[110:111], v74 offset1:1
	ds_load_2addr_b32 v[108:109], v74 offset0:2 offset1:3
	ds_load_2addr_b64 v[214:217], v72 offset1:32
	ds_load_2addr_b64 v[218:221], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[104:105], v74 offset0:4 offset1:5
	ds_load_2addr_b32 v[106:107], v74 offset0:6 offset1:7
	ds_load_2addr_b32 v[102:103], v74 offset0:8 offset1:9
	ds_load_2addr_b32 v[98:99], v74 offset0:10 offset1:11
	ds_load_2addr_b32 v[100:101], v74 offset0:12 offset1:13
	ds_load_2addr_b32 v[96:97], v74 offset0:14 offset1:15
	ds_load_2addr_b32 v[90:91], v74 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v74 offset0:34 offset1:35
	ds_load_2addr_b32 v[84:85], v74 offset0:36 offset1:37
	ds_load_2addr_b32 v[82:83], v74 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v74 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v74 offset0:42 offset1:43
	ds_load_2addr_b32 v[76:77], v74 offset0:44 offset1:45
	ds_load_2addr_b32 v[74:75], v74 offset0:46 offset1:47
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0xf
	v_wmma_i32_16x16x32_iu4 v[57:64], v[214:215], v[198:199], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[214:215], v[202:203], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[214:215], v[206:207], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[214:215], v[210:211], v[33:40] neg_lo:[0,1,0]
	s_wait_dscnt 0xe
	v_wmma_i32_16x16x32_iu4 v[25:32], v[218:219], v[198:199], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[218:219], v[202:203], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[218:219], v[206:207], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[218:219], v[210:211], v[1:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[216:217], v[200:201], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[216:217], v[204:205], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[216:217], v[208:209], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[216:217], v[212:213], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[220:221], v[200:201], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[220:221], v[204:205], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[220:221], v[208:209], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[220:221], v[212:213], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v73, s33, v0
	ds_load_2addr_b32 v[94:95], v73 offset1:1
	ds_load_2addr_b32 v[92:93], v73 offset0:32 offset1:33
	ds_load_2addr_b32 v[88:89], v73 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v73 offset0:96 offset1:97
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_10
; %bb.14:                               ; %.preheader469.i
                                        ;   in Loop: Header=BB1_11 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v0, off, off offset:84
	scratch_load_b32 v65, off, off offset:68
	s_and_b32 s4, s31, exec_lo
	s_cselect_b32 s4, s25, 0x3400
	s_cselect_b32 s30, s26, 0x3c00
	v_cndmask_b32_e64 v119, 0, v119, s0
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v0, v65
	scratch_load_b32 v0, off, off offset:72 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v198, v198.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v199, 0, v0, s2
	scratch_load_b32 v0, off, off offset:108 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s4, v0
	v_add_nc_u32_e32 v201, s30, v0
	scratch_load_b32 v0, off, off offset:52 ; 4-byte Folded Reload
	s_movk_i32 s30, 0x1000
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:8
	scratch_load_b32 v0, off, off offset:60 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB1_10
.LBB1_15:
	s_mov_b32 s24, -1
.LBB1_16:                               ; %Flow1912
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s24
	s_cbranch_vccnz .LBB1_108
	s_branch .LBB1_252
.LBB1_17:                               ; %Flow
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v17, off, off offset:112
	scratch_load_b32 v18, off, off offset:116
	scratch_load_b32 v15, off, off offset:120
	scratch_load_b32 v0, off, off offset:124
.LBB1_18:                               ; %.preheader466.i
	s_wait_loadcnt 0x3
	v_lshrrev_b32_e32 v2, 4, v17
	s_wait_loadcnt 0x2
	v_mul_u32_u24_e32 v1, 0x500, v18
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v5, 15, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add3_u32 v4, 0, v1, v3
	v_or_b32_e32 v1, s22, v0
	v_mul_u32_u24_e32 v3, 0x50, v0
	v_lshlrev_b32_e32 v6, 2, v5
	v_or_b32_e32 v5, s11, v5
	v_mad_u32_u24 v7, 0x280, v15, v4
	v_cmp_gt_i32_e64 s7, s12, v1
	v_ashrrev_i32_e32 v2, 31, v1
	v_add3_u32 v3, 0, v3, v6
	v_cmp_gt_i32_e32 vcc_lo, s14, v5
	ds_store_2addr_b32 v7, v179, v186 offset1:20
	ds_store_2addr_b32 v7, v185, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v7, v182, v183 offset0:80 offset1:100
	ds_store_2addr_b32 v7, v180, v181 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_20
; %bb.19:
	v_mad_co_i64_i32 v[6:7], null, s12, v5, 0
	v_lshlrev_b64_e32 v[8:9], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[6:7], 2, v[6:7]
	v_add_co_u32 v0, s0, s20, v6
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, s21, v7, s0
	v_add_co_u32 v6, s0, v0, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v7, null, v7, v9, s0
	ds_load_b32 v8, v3
	global_load_b32 v0, v[6:7], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v8, v0
	global_store_b32 v[6:7], v0, off
.LBB1_20:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v6, 64, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v6
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_22
; %bb.21:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:1280
	global_load_b32 v0, v[7:8], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off
.LBB1_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v0
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_24
; %bb.23:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:2560
	global_load_b32 v0, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off offset:128
.LBB1_24:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_26
; %bb.25:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:3840
	global_load_b32 v0, v[7:8], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off offset:128
.LBB1_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v0
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_28
; %bb.27:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:5120
	global_load_b32 v0, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off offset:256
.LBB1_28:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_30
; %bb.29:
	v_mad_co_i64_i32 v[7:8], null, s12, v6, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:6400
	global_load_b32 v0, v[7:8], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off offset:256
.LBB1_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v0, 0x60, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v0
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_32
; %bb.31:
	v_mad_co_i64_i32 v[7:8], null, s12, v5, 0
	v_lshlrev_b64_e32 v[9:10], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[7:8], 2, v[7:8]
	v_add_co_u32 v0, s1, s20, v7
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s21, v8, s1
	v_add_co_u32 v7, s1, v0, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, v8, v10, s1
	ds_load_b32 v9, v3 offset:7680
	global_load_b32 v0, v[7:8], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v9, v0
	global_store_b32 v[7:8], v0, off offset:384
.LBB1_32:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v7, 0x280, v15
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_34
; %bb.33:
	v_mad_co_i64_i32 v[8:9], null, s12, v6, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v0, s1, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s1
	v_add_co_u32 v8, s1, v0, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s1
	ds_load_b32 v10, v3 offset:8960
	global_load_b32 v0, v[8:9], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v10, v0
	global_store_b32 v[8:9], v0, off offset:384
.LBB1_34:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v4, v4, v7
	v_or_b32_e32 v7, 16, v5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s1, s14, v7
	ds_store_2addr_b32 v4, v176, v177 offset1:20
	ds_store_2addr_b32 v4, v174, v175 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v172, v173 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v170, v171 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_36
; %bb.35:
	v_mad_co_i64_i32 v[8:9], null, s12, v7, 0
	v_lshlrev_b64_e32 v[10:11], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[8:9], 2, v[8:9]
	v_add_co_u32 v0, s2, s20, v8
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, s21, v9, s2
	v_add_co_u32 v8, s2, v0, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v9, null, v9, v11, s2
	ds_load_b32 v10, v3
	global_load_b32 v0, v[8:9], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v10, v0
	global_store_b32 v[8:9], v0, off
.LBB1_36:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v8, 0x50, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v8
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
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:8960
	global_load_b32 v0, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:384
.LBB1_44:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v9, 32, v5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s3, s14, v9
	ds_store_2addr_b32 v4, v167, v168 offset1:20
	ds_store_2addr_b32 v4, v165, v166 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v163, v164 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v161, v162 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_46
; %bb.45:
	v_mad_co_i64_i32 v[10:11], null, s12, v9, 0
	v_lshlrev_b64_e32 v[12:13], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[10:11], 2, v[10:11]
	v_add_co_u32 v0, s4, s20, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s21, v11, s4
	v_add_co_u32 v10, s4, v0, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, v11, v13, s4
	ds_load_b32 v12, v3
	global_load_b32 v0, v[10:11], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v12, v0
	global_store_b32 v[10:11], v0, off
.LBB1_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v10, 0x60, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v10
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
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:8960
	global_load_b32 v0, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:384
.LBB1_54:                               ; %.preheader.3.i
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v11, 48, v5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s5, s14, v11
	ds_store_2addr_b32 v4, v159, v160 offset1:20
	ds_store_2addr_b32 v4, v157, v158 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v155, v156 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v153, v154 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s15, s6
	s_cbranch_execz .LBB1_56
; %bb.55:
	v_mad_co_i64_i32 v[12:13], null, s12, v11, 0
	v_lshlrev_b64_e32 v[14:15], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[12:13], 2, v[12:13]
	v_add_co_u32 v0, s6, s20, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s21, v13, s6
	v_add_co_u32 v12, s6, v0, v14
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, v13, v15, s6
	ds_load_b32 v14, v3
	global_load_b32 v0, v[12:13], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v14, v0
	global_store_b32 v[12:13], v0, off
.LBB1_56:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v12, 0x70, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v12
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
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:8960
	global_load_b32 v0, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:384
.LBB1_64:                               ; %.preheader465.1.i
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v0, 16, v1
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s7, s12, v0
	ds_store_2addr_b32 v4, v152, v151 offset1:20
	ds_store_2addr_b32 v4, v149, v150 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v148, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v145, v146 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s8, s20, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s8
	v_add_co_u32 v13, s8, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s8
	ds_load_b32 v15, v3
	global_load_b32 v0, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:64
.LBB1_66:
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_68
; %bb.67:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s8, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s8
	v_add_co_u32 v13, s8, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s8
	ds_load_b32 v15, v3 offset:1280
	global_load_b32 v0, v[13:14], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:64
.LBB1_68:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v0, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v0
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_70
; %bb.69:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s9, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s9
	v_add_co_u32 v13, s9, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s9
	ds_load_b32 v15, v3 offset:2560
	global_load_b32 v0, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:192
.LBB1_70:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_72
; %bb.71:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s9, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s9
	v_add_co_u32 v13, s9, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s9
	ds_load_b32 v15, v3 offset:3840
	global_load_b32 v0, v[13:14], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:192
.LBB1_72:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v0, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v0
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_74
; %bb.73:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s10, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s10
	v_add_co_u32 v13, s10, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s10
	ds_load_b32 v15, v3 offset:5120
	global_load_b32 v0, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:320
.LBB1_74:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s10
	s_cbranch_execz .LBB1_76
; %bb.75:
	v_mad_co_i64_i32 v[13:14], null, s12, v6, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s10, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s10
	v_add_co_u32 v13, s10, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s10
	ds_load_b32 v15, v3 offset:6400
	global_load_b32 v0, v[13:14], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:320
.LBB1_76:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	v_or_b32_e32 v0, 0x70, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s10, s12, v0
	s_and_b32 s25, s10, vcc_lo
	s_and_saveexec_b32 s15, s25
	s_cbranch_execz .LBB1_78
; %bb.77:
	v_mad_co_i64_i32 v[13:14], null, s12, v5, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, vcc_lo, s20, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v14, vcc_lo
	v_add_co_u32 v13, vcc_lo, v0, v15
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v5, v16, vcc_lo
	ds_load_b32 v5, v3 offset:7680
	global_load_b32 v0, v[13:14], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v5, v0
	global_store_b32 v[13:14], v0, off offset:448
.LBB1_78:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s15, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_80
; %bb.79:
	v_mad_co_i64_i32 v[5:6], null, s12, v6, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:8960
	global_load_b32 v0, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:448
.LBB1_80:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s1
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v143, v144 offset1:20
	ds_store_2addr_b32 v4, v141, v142 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v139, v140 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v137, v138 offset0:120 offset1:140
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
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:8960
	global_load_b32 v0, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:448
.LBB1_89:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v135, v136 offset1:20
	ds_store_2addr_b32 v4, v133, v134 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v28, v132 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v26, v27 offset0:120 offset1:140
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
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:8960
	global_load_b32 v0, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:448
.LBB1_98:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v4, v29, v25 offset1:20
	ds_store_2addr_b32 v4, v23, v24 offset0:40 offset1:60
	ds_store_2addr_b32 v4, v21, v22 offset0:80 offset1:100
	ds_store_2addr_b32 v4, v20, v19 offset0:120 offset1:140
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
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[1:2], 2, v[1:2]
	ds_load_b32 v3, v3 offset:8960
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, s21, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, v4, v2, vcc_lo
	global_load_b32 v0, v[1:2], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v3, v0
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
; %bb.109:                              ; %.preheader476.i18
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
	s_mul_i32 s4, s10, 0x88
	v_cmpx_gt_u32_e32 0x100, v17
	s_cbranch_execz .LBB1_113
; %bb.110:                              ; %.lr.ph.i298
	v_lshrrev_b32_e32 v1, 1, v17
	v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v3, 1, v17
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v5, s11, v1
	v_lshlrev_b32_e32 v2, 2, v3
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_gt_i32_e64 s14, v5
	s_cbranch_execz .LBB1_112
; %bb.111:
	v_mad_co_i64_i32 v[4:5], null, 0x48, v5, s[18:19]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, v4, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	global_load_b32 v4, v[4:5], off
.LBB1_112:                              ; %.preheader471.loopexit.i299
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v0, s22, v1
	v_lshlrev_b32_e32 v1, 3, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_i32_e32 v5, s2, v0
	v_cmp_gt_i32_e32 vcc_lo, s12, v0
	v_lshlrev_b32_e32 v3, 4, v3
	v_add3_u32 v1, 0, v1, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_mad_co_i64_i32 v[5:6], null, s4, v5, s[16:17]
	global_load_b32 v5, v[5:6], off
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v3, v3, v5
	v_cvt_f32_f16_e32 v0, v3.l
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0, v0, vcc_lo
	ds_store_2addr_stride64_b32 v1, v4, v0 offset0:48 offset1:56
.LBB1_113:                              ; %Flow1911
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v0, 2, v17
	v_dual_mov_b32 v68, 0 :: v_dual_and_b32 v1, 3, v17
	s_add_co_i32 s3, s14, -1
	v_dual_mov_b32 v133, 0 :: v_dual_lshlrev_b32 v12, 4, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v66, 0 :: v_dual_add_nc_u32 v9, s11, v0
	v_dual_mov_b32 v65, 0 :: v_dual_add_nc_u32 v2, s22, v0
	v_dual_mov_b32 v70, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v67, 0 :: v_dual_add_nc_u32 v10, 64, v9
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v3, s3, v9
	v_dual_mov_b32 v69, 0 :: v_dual_add_nc_u32 v4, 64, v2
	v_min_i32_e32 v2, s2, v2
	v_min_i32_e32 v5, s3, v10
	v_dual_mov_b32 v147, 0 :: v_dual_and_b32 v12, 16, v12
	v_bfe_u32 v11, v17, 1, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[109:110], null, 0x48, v3, v[1:2]
	v_min_i32_e32 v3, s2, v4
	v_mad_co_u64_u32 v[110:111], null, 0x48, v5, v[1:2]
	v_mad_co_u64_u32 v[111:112], null, s4, v2, v[1:2]
	v_or_b32_e32 v13, 8, v18
	v_and_or_b32 v0, v0, 15, v12
	v_mad_co_u64_u32 v[120:121], null, s4, v3, v[1:2]
	s_clause 0x1
	global_load_b64 v[1:2], v109, s[18:19] offset:8
	global_load_b64 v[3:4], v110, s[18:19] offset:8
	s_clause 0x1
	global_load_b64 v[5:6], v111, s[16:17] offset:8
	global_load_b64 v[7:8], v120, s[16:17] offset:8
	v_and_or_b32 v12, v18, 6, v11
	v_dual_mov_b32 v185, 0 :: v_dual_lshlrev_b32 v0, 3, v0
	v_and_or_b32 v11, v13, 14, v11
	v_cmp_gt_i32_e64 s0, s14, v9
	v_cmp_gt_i32_e64 s1, s14, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshl_or_b32 v12, v12, 8, v0
	v_mov_b32_e32 v182, 0
	v_lshl_or_b32 v0, v11, 8, v0
	v_dual_mov_b32 v100, 0 :: v_dual_and_b32 v15, 15, v17
	v_add_nc_u32_e32 v9, 0, v12
	v_bfe_u32 v14, v17, 4, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v0, 0, v0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v153, 0
	scratch_store_b32 off, v9, off offset:28 ; 4-byte Folded Spill
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v155, 0
	scratch_store_b32 off, v0, off offset:32 ; 4-byte Folded Spill
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v127, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v168, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v143, 0
	v_dual_mov_b32 v171, 0 :: v_dual_mov_b32 v170, 0
	v_dual_mov_b32 v173, 0 :: v_dual_mov_b32 v172, 0
	v_dual_mov_b32 v175, 0 :: v_dual_mov_b32 v174, 0
	v_dual_mov_b32 v177, 0 :: v_dual_mov_b32 v176, 0
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v149, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v181, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v183, 0
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v179, 0
	v_mov_b32_e32 v186, 0
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v2, 0, v2, s0
	v_cndmask_b32_e64 v1, 0, v1, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v4, 0, v4, s1
	v_cndmask_b32_e64 v3, 0, v3, s1
	s_wait_loadcnt 0x1
	ds_store_2addr_stride64_b64 v9, v[1:2], v[5:6] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[3:4], v[7:8] offset1:8
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_162
; %bb.114:                              ; %.preheader470.lr.ph.i191
	v_dual_mov_b32 v7, 0 :: v_dual_and_b32 v4, 1, v17
	v_dual_mov_b32 v179, 0 :: v_dual_and_b32 v0, 31, v17
	v_bfe_u32 v2, v17, 5, 1
	v_lshrrev_b32_e32 v3, 1, v17
	s_clause 0x2                            ; 12-byte Folded Spill
	scratch_store_b32 off, v17, off offset:112
	scratch_store_b32 off, v7, off offset:72
	scratch_store_b32 off, v15, off offset:124
	v_dual_mov_b32 v7, 0 :: v_dual_lshlrev_b32 v6, 3, v15
	v_lshrrev_b32_e32 v1, 6, v17
	v_lshlrev_b32_e32 v0, 3, v0
	v_lshl_add_u32 v11, v2, 11, 0
	v_lshlrev_b32_e32 v13, 4, v4
	s_clause 0x1                            ; 8-byte Folded Spill
	scratch_store_b32 off, v7, off offset:68
	scratch_store_b32 off, v14, off offset:120
	v_add_nc_u32_e32 v7, s22, v3
	v_lshl_or_b32 v121, v1, 10, v0
	v_dual_mov_b32 v0, v109 :: v_dual_add_nc_u32 v197, v11, v0
	v_dual_mov_b32 v186, 0 :: v_dual_lshlrev_b32 v5, 6, v14
	s_delay_alu instid0(VALU_DEP_4)
	v_min_i32_e32 v10, s2, v7
	scratch_store_b32 off, v13, off offset:84 ; 4-byte Folded Spill
	v_add_nc_u32_e32 v8, s11, v3
	v_lshlrev_b32_e32 v3, 3, v3
	v_lshlrev_b32_e32 v9, 2, v4
	v_mad_co_u64_u32 v[13:14], null, s4, v10, s[16:17]
	v_lshlrev_b32_e32 v12, 8, v1
	v_ashrrev_i32_e32 v1, 31, v10
	v_lshlrev_b32_e32 v2, 9, v2
	v_add3_u32 v3, 0, v3, v9
	v_dual_mov_b32 v185, 0 :: v_dual_lshlrev_b32 v4, 2, v4
	v_mov_b32_e32 v184, 0
	v_mad_co_u64_u32 v[14:15], null, s4, v1, v[14:15]
	v_add3_u32 v1, 0, v12, v5
	scratch_store_b32 off, v3, off offset:108 ; 4-byte Folded Spill
	v_dual_mov_b32 v182, 0 :: v_dual_mov_b32 v183, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v181, 0
	scratch_store_b32 off, v1, off offset:76 ; 4-byte Folded Spill
	v_add3_u32 v1, 0, v2, v6
	scratch_store_b32 off, v18, off offset:116 ; 4-byte Folded Spill
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v151, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v1, off offset:80
	scratch_store_b64 off, v[0:1], off offset:36
	v_mov_b32_e32 v0, v110
	scratch_store_b96 off, v[13:15], off offset:88 ; 12-byte Folded Spill
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v147, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v146, 0
	scratch_store_b64 off, v[0:1], off offset:44 ; 8-byte Folded Spill
	v_mov_b32_e32 v0, v111
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v177, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v168, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v159, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v156, 0
	v_dual_mov_b32 v153, 0 :: v_dual_mov_b32 v154, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v65, 0
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v4, off offset:100
	scratch_store_b64 off, v[0:1], off offset:52
	v_min_i32_e32 v4, s3, v8
	v_cmp_gt_i32_e64 s2, s14, v8
	v_cmp_gt_i32_e64 s3, s12, v7
	v_mov_b32_e32 v0, v120
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s27, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s6, s5
	s_clause 0x1                            ; 12-byte Folded Spill
	scratch_store_b32 off, v4, off offset:104
	scratch_store_b64 off, v[0:1], off offset:60
	s_branch .LBB1_116
.LBB1_115:                              ;   in Loop: Header=BB1_116 Depth=1
	s_and_b32 vcc_lo, exec_lo, s7
	s_mov_b32 s6, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_161
.LBB1_116:                              ; %.preheader470.i196
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_118 Depth 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s24, s6, 1
	s_mov_b32 s7, s5
	s_lshl_b32 s25, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[8:9], s[6:7], 0x88
	s_cmp_eq_u32 s24, s10
	s_mov_b32 s29, 0
	s_mov_b32 s26, -1
	s_mov_b32 s28, 0
	s_cselect_b32 s7, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[8:9], s[16:17], s[8:9]
	s_branch .LBB1_118
.LBB1_117:                              ;   in Loop: Header=BB1_118 Depth=2
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v114, v57
	v_cvt_f32_i32_e32 v115, v58
	v_mul_f32_e32 v199, v98, v94
	v_cvt_f32_i32_e32 v120, v62
	v_mul_f32_e32 v208, v106, v92
	v_cvt_f32_i32_e32 v210, v52
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_mul_f32_e32 v67, v88, v90
	v_mul_f32_e32 v196, v88, v86
	v_mul_f32_e32 v204, v108, v92
	v_mul_f32_e32 v218, v108, v88
	v_cvt_f32_i32_e32 v34, v34
	v_mul_f32_e32 v108, v108, v72
	v_mul_f32_e32 v203, v110, v92
	v_mul_f32_e32 v217, v110, v88
	v_cvt_f32_i32_e32 v33, v33
	v_dual_mul_f32 v110, v110, v72 :: v_dual_mul_f32 v57, v111, v94
	v_mul_f32_e32 v58, v109, v94
	v_dual_mul_f32 v116, v104, v94 :: v_dual_mul_f32 v117, v106, v94
	v_cvt_f32_i32_e32 v118, v59
	v_cvt_f32_i32_e32 v119, v60
	v_dual_mul_f32 v59, v105, v94 :: v_dual_mul_f32 v60, v107, v94
	v_mul_f32_e32 v198, v102, v94
	v_cvt_f32_i32_e32 v0, v61
	v_mul_f32_e32 v61, v103, v94
	v_mul_f32_e32 v62, v99, v94
	v_mul_f32_e32 v200, v100, v94
	v_mul_f32_e32 v201, v96, v94
	v_cvt_f32_i32_e32 v202, v63
	v_mul_f32_e32 v63, v101, v94
	v_cvt_f32_i32_e32 v205, v49
	v_cvt_f32_i32_e32 v206, v50
	v_dual_mul_f32 v49, v111, v92 :: v_dual_mul_f32 v50, v109, v92
	v_mul_f32_e32 v207, v104, v92
	v_cvt_f32_i32_e32 v209, v51
	v_dual_mul_f32 v51, v105, v92 :: v_dual_mul_f32 v52, v107, v92
	v_mul_f32_e32 v211, v102, v92
	v_mul_f32_e32 v212, v98, v92
	v_cvt_f32_i32_e32 v213, v53
	v_cvt_f32_i32_e32 v214, v54
	v_mul_f32_e32 v53, v103, v92
	v_dual_mul_f32 v54, v99, v92 :: v_dual_mul_f32 v215, v100, v92
	v_cvt_f32_i32_e32 v216, v55
	v_mul_f32_e32 v55, v101, v92
	v_cvt_f32_i32_e32 v219, v41
	v_cvt_f32_i32_e32 v220, v42
	v_dual_mul_f32 v41, v111, v88 :: v_dual_mul_f32 v42, v109, v88
	v_dual_mul_f32 v221, v104, v88 :: v_dual_mul_f32 v222, v106, v88
	v_dual_mul_f32 v223, v105, v88 :: v_dual_mul_f32 v224, v107, v88
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v11, v11
	v_mul_f32_e32 v225, v102, v88
	v_dual_mul_f32 v226, v98, v88 :: v_dual_mul_f32 v227, v103, v88
	v_dual_mul_f32 v228, v99, v88 :: v_dual_mul_f32 v229, v100, v88
	v_mul_f32_e32 v230, v101, v88
	v_mul_f32_e32 v111, v111, v72
	v_dual_mul_f32 v109, v109, v72 :: v_dual_mul_f32 v104, v104, v72
	v_dual_mul_f32 v105, v105, v72 :: v_dual_mul_f32 v106, v106, v72
	v_dual_mul_f32 v107, v107, v72 :: v_dual_mul_f32 v102, v102, v72
	v_dual_mul_f32 v103, v103, v72 :: v_dual_mul_f32 v98, v98, v72
	v_dual_mul_f32 v99, v99, v72 :: v_dual_mul_f32 v100, v100, v72
	v_mul_f32_e32 v101, v101, v72
	v_dual_mul_f32 v231, v97, v94 :: v_dual_mul_f32 v232, v96, v92
	v_mul_f32_e32 v233, v97, v92
	v_dual_mul_f32 v234, v96, v88 :: v_dual_mul_f32 v235, v97, v88
	v_dual_mul_f32 v96, v96, v72 :: v_dual_mul_f32 v97, v97, v72
	v_cvt_f32_i32_e32 v95, v95
	v_mul_f32_e32 v236, v94, v90
	v_dual_mul_f32 v237, v94, v86 :: v_dual_mul_f32 v238, v94, v91
	v_dual_mul_f32 v239, v94, v87 :: v_dual_mul_f32 v240, v94, v84
	v_dual_mul_f32 v241, v94, v82 :: v_dual_mul_f32 v242, v94, v85
	v_dual_mul_f32 v243, v94, v83 :: v_dual_mul_f32 v244, v94, v80
	v_dual_mul_f32 v245, v94, v78 :: v_dual_mul_f32 v246, v94, v81
	v_dual_mul_f32 v247, v94, v79 :: v_dual_mul_f32 v248, v94, v76
	v_dual_mul_f32 v249, v94, v74 :: v_dual_mul_f32 v250, v94, v77
	v_mul_f32_e32 v94, v94, v75
	v_cvt_f32_i32_e32 v93, v93
	v_mul_f32_e32 v251, v92, v90
	v_dual_mul_f32 v252, v92, v86 :: v_dual_mul_f32 v253, v92, v91
	v_dual_mul_f32 v254, v92, v87 :: v_dual_mul_f32 v255, v92, v84
	v_dual_mul_f32 v169, v92, v82 :: v_dual_mul_f32 v178, v92, v85
	v_mul_f32_e32 v189, v92, v83
	v_dual_mul_f32 v191, v92, v80 :: v_dual_mul_f32 v194, v92, v78
	v_mul_f32_e32 v190, v92, v81
	v_dual_mul_f32 v192, v92, v79 :: v_dual_mul_f32 v65, v92, v76
	v_dual_mul_f32 v66, v92, v74 :: v_dual_mul_f32 v195, v92, v77
	v_mul_f32_e32 v92, v92, v75
	v_cvt_f32_i32_e32 v89, v89
	v_mul_f32_e32 v68, v88, v91
	v_dual_mul_f32 v69, v88, v87 :: v_dual_mul_f32 v70, v88, v84
	v_mul_f32_e32 v71, v88, v82
	v_dual_mul_f32 v187, v88, v85 :: v_dual_mul_f32 v188, v88, v83
	v_mul_f32_e32 v193, v88, v80
	v_dual_mul_f32 v121, v88, v78 :: v_dual_mul_f32 v122, v88, v81
	v_dual_mul_f32 v123, v88, v79 :: v_dual_mul_f32 v124, v88, v76
	v_dual_mul_f32 v125, v88, v74 :: v_dual_mul_f32 v126, v88, v77
	v_mul_f32_e32 v88, v88, v75
	v_dual_mul_f32 v90, v72, v90 :: v_dual_mul_f32 v91, v72, v91
	v_dual_mul_f32 v86, v72, v86 :: v_dual_mul_f32 v87, v72, v87
	v_dual_mul_f32 v84, v72, v84 :: v_dual_mul_f32 v85, v72, v85
	v_dual_mul_f32 v82, v72, v82 :: v_dual_mul_f32 v83, v72, v83
	v_dual_mul_f32 v80, v72, v80 :: v_dual_mul_f32 v81, v72, v81
	v_dual_mul_f32 v78, v72, v78 :: v_dual_mul_f32 v79, v72, v79
	v_dual_mul_f32 v76, v72, v76 :: v_dual_mul_f32 v77, v72, v77
	v_mul_f32_e32 v74, v72, v74
	v_mul_f32_e32 v72, v72, v75
	v_dual_mul_f32 v75, v112, v114 :: v_dual_mul_f32 v112, v113, v115
	v_dual_mul_f32 v115, v199, v120 :: v_dual_mul_f32 v120, v208, v210
	v_dual_mul_f32 v9, v67, v9 :: v_dual_mul_f32 v10, v196, v10
	v_cvt_f32_i32_e32 v73, v73
	v_dual_mul_f32 v34, v108, v34 :: v_dual_mul_f32 v33, v110, v33
	v_cvt_f32_i32_e32 v14, v14
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_mul_f32_e32 v39, v100, v39
	v_dual_mul_f32 v23, v65, v23 :: v_dual_mul_f32 v24, v66, v24
	v_mul_f32_e32 v11, v70, v11
	v_fmac_f32_e32 v112, v58, v95
	v_fmac_f32_e32 v120, v52, v93
	v_dual_fmac_f32 v9, v68, v89 :: v_dual_fmac_f32 v10, v69, v89
	s_clause 0x6                            ; 28-byte Folded Reload
	scratch_load_b32 v100, off, off offset:24 th:TH_LOAD_LU
	scratch_load_b32 v69, off, off offset:20 th:TH_LOAD_LU
	scratch_load_b32 v70, off, off offset:16 th:TH_LOAD_LU
	scratch_load_b32 v67, off, off offset:12 th:TH_LOAD_LU
	scratch_load_b32 v68, off, off offset:8 th:TH_LOAD_LU
	scratch_load_b32 v66, off, off offset:4 th:TH_LOAD_LU
	scratch_load_b32 v65, off, off th:TH_LOAD_LU
	v_fmac_f32_e32 v34, v109, v73
	scratch_load_b64 v[109:110], off, off offset:36 ; 8-byte Folded Reload
	v_fmac_f32_e32 v33, v111, v73
	s_wait_loadcnt 0x0
	scratch_load_b64 v[110:111], off, off offset:44 ; 8-byte Folded Reload
	v_mul_f32_e32 v14, v121, v14
	v_add_f32_e32 v186, v186, v112
	v_add_f32_e32 v175, v175, v120
	s_wait_loadcnt 0x0
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b64 v[111:112], off, off offset:52
	scratch_load_b64 v[120:121], off, off offset:60
	v_cvt_f32_i32_e32 v64, v64
	v_cvt_f32_i32_e32 v56, v56
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v15, v15
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_dual_mul_f32 v113, v116, v118 :: v_dual_mul_f32 v114, v117, v119
	v_mul_f32_e32 v0, v198, v0
	v_mul_f32_e32 v116, v200, v202
	v_dual_mul_f32 v64, v201, v64 :: v_dual_mul_f32 v117, v203, v205
	v_dual_mul_f32 v118, v204, v206 :: v_dual_mul_f32 v119, v207, v209
	v_dual_mul_f32 v198, v211, v213 :: v_dual_mul_f32 v199, v212, v214
	v_mul_f32_e32 v200, v215, v216
	v_dual_mul_f32 v56, v232, v56 :: v_dual_mul_f32 v201, v217, v219
	v_dual_mul_f32 v202, v218, v220 :: v_dual_mul_f32 v43, v221, v43
	v_dual_mul_f32 v44, v222, v44 :: v_dual_mul_f32 v45, v225, v45
	v_dual_mul_f32 v46, v226, v46 :: v_dual_mul_f32 v47, v229, v47
	v_dual_mul_f32 v48, v234, v48 :: v_dual_mul_f32 v35, v104, v35
	v_mul_f32_e32 v36, v106, v36
	v_mul_f32_e32 v37, v102, v37
	v_mul_f32_e32 v38, v98, v38
	v_mul_f32_e32 v40, v96, v40
	v_dual_mul_f32 v25, v236, v25 :: v_dual_mul_f32 v26, v237, v26
	v_dual_mul_f32 v27, v240, v27 :: v_dual_mul_f32 v28, v241, v28
	v_dual_mul_f32 v29, v244, v29 :: v_dual_mul_f32 v30, v245, v30
	v_dual_mul_f32 v31, v248, v31 :: v_dual_mul_f32 v32, v249, v32
	v_dual_mul_f32 v17, v251, v17 :: v_dual_mul_f32 v18, v252, v18
	v_dual_mul_f32 v19, v255, v19 :: v_dual_mul_f32 v20, v169, v20
	v_dual_mul_f32 v21, v191, v21 :: v_dual_mul_f32 v22, v194, v22
	v_dual_mul_f32 v12, v71, v12 :: v_dual_mul_f32 v13, v193, v13
	v_dual_mul_f32 v15, v124, v15 :: v_dual_mul_f32 v16, v125, v16
	v_mul_f32_e32 v1, v90, v1
	v_dual_mul_f32 v2, v86, v2 :: v_dual_mul_f32 v3, v84, v3
	v_dual_mul_f32 v4, v82, v4 :: v_dual_mul_f32 v5, v80, v5
	v_dual_mul_f32 v6, v78, v6 :: v_dual_mul_f32 v7, v76, v7
	v_dual_mul_f32 v8, v74, v8 :: v_dual_fmac_f32 v75, v57, v95
	v_dual_fmac_f32 v113, v59, v95 :: v_dual_fmac_f32 v114, v60, v95
	v_dual_fmac_f32 v0, v61, v95 :: v_dual_fmac_f32 v115, v62, v95
	v_fmac_f32_e32 v116, v63, v95
	v_dual_fmac_f32 v64, v231, v95 :: v_dual_fmac_f32 v117, v49, v93
	v_dual_fmac_f32 v118, v50, v93 :: v_dual_fmac_f32 v119, v51, v93
	v_dual_fmac_f32 v198, v53, v93 :: v_dual_fmac_f32 v199, v54, v93
	v_fmac_f32_e32 v200, v55, v93
	v_fmac_f32_e32 v56, v233, v93
	v_dual_fmac_f32 v201, v41, v89 :: v_dual_fmac_f32 v202, v42, v89
	v_dual_fmac_f32 v43, v223, v89 :: v_dual_fmac_f32 v44, v224, v89
	v_dual_fmac_f32 v45, v227, v89 :: v_dual_fmac_f32 v46, v228, v89
	v_dual_fmac_f32 v47, v230, v89 :: v_dual_fmac_f32 v48, v235, v89
	v_dual_fmac_f32 v35, v105, v73 :: v_dual_fmac_f32 v36, v107, v73
	v_fmac_f32_e32 v37, v103, v73
	v_dual_fmac_f32 v38, v99, v73 :: v_dual_fmac_f32 v39, v101, v73
	v_dual_fmac_f32 v40, v97, v73 :: v_dual_fmac_f32 v25, v238, v95
	v_dual_fmac_f32 v26, v239, v95 :: v_dual_fmac_f32 v27, v242, v95
	v_dual_fmac_f32 v28, v243, v95 :: v_dual_fmac_f32 v29, v246, v95
	v_dual_fmac_f32 v30, v247, v95 :: v_dual_fmac_f32 v31, v250, v95
	v_dual_fmac_f32 v32, v94, v95 :: v_dual_fmac_f32 v17, v253, v93
	v_fmac_f32_e32 v18, v254, v93
	v_dual_fmac_f32 v19, v178, v93 :: v_dual_fmac_f32 v20, v189, v93
	v_dual_fmac_f32 v21, v190, v93 :: v_dual_fmac_f32 v22, v192, v93
	v_dual_fmac_f32 v23, v195, v93 :: v_dual_fmac_f32 v24, v92, v93
	v_dual_fmac_f32 v11, v187, v89 :: v_dual_fmac_f32 v12, v188, v89
	v_dual_fmac_f32 v13, v122, v89 :: v_dual_fmac_f32 v14, v123, v89
	v_dual_fmac_f32 v15, v126, v89 :: v_dual_fmac_f32 v16, v88, v89
	v_fmac_f32_e32 v1, v91, v73
	v_dual_fmac_f32 v2, v87, v73 :: v_dual_fmac_f32 v3, v85, v73
	v_dual_fmac_f32 v4, v83, v73 :: v_dual_fmac_f32 v5, v81, v73
	v_dual_fmac_f32 v6, v79, v73 :: v_dual_fmac_f32 v7, v77, v73
	v_dual_fmac_f32 v8, v72, v73 :: v_dual_add_f32 v179, v179, v75
	v_dual_add_f32 v185, v185, v113 :: v_dual_add_f32 v184, v184, v114
	v_dual_add_f32 v182, v182, v0 :: v_dual_add_f32 v183, v183, v115
	v_add_f32_e32 v180, v180, v116
	v_dual_add_f32 v181, v181, v64 :: v_dual_add_f32 v176, v176, v117
	v_dual_add_f32 v177, v177, v118 :: v_dual_add_f32 v174, v174, v119
	v_dual_add_f32 v172, v172, v198 :: v_dual_add_f32 v173, v173, v199
	v_add_f32_e32 v170, v170, v200
	v_add_f32_e32 v171, v171, v56
	v_dual_add_f32 v167, v167, v201 :: v_dual_add_f32 v168, v168, v202
	v_dual_add_f32 v165, v165, v43 :: v_dual_add_f32 v166, v166, v44
	v_dual_add_f32 v163, v163, v45 :: v_dual_add_f32 v164, v164, v46
	v_dual_add_f32 v161, v161, v47 :: v_dual_add_f32 v162, v162, v48
	v_dual_add_f32 v159, v159, v33 :: v_dual_add_f32 v160, v160, v34
	v_dual_add_f32 v157, v157, v35 :: v_dual_add_f32 v158, v158, v36
	v_dual_add_f32 v155, v155, v37 :: v_dual_add_f32 v156, v156, v38
	v_dual_add_f32 v153, v153, v39 :: v_dual_add_f32 v154, v154, v40
	v_dual_add_f32 v152, v152, v25 :: v_dual_add_f32 v151, v151, v26
	v_dual_add_f32 v149, v149, v27 :: v_dual_add_f32 v150, v150, v28
	v_dual_add_f32 v148, v148, v29 :: v_dual_add_f32 v147, v147, v30
	v_dual_add_f32 v145, v145, v31 :: v_dual_add_f32 v146, v146, v32
	v_dual_add_f32 v143, v143, v17 :: v_dual_add_f32 v144, v144, v18
	v_dual_add_f32 v141, v141, v19 :: v_dual_add_f32 v142, v142, v20
	v_dual_add_f32 v139, v139, v21 :: v_dual_add_f32 v140, v140, v22
	v_dual_add_f32 v137, v137, v23 :: v_dual_add_f32 v138, v138, v24
	v_dual_add_f32 v132, v132, v9 :: v_dual_add_f32 v131, v131, v10
	v_dual_add_f32 v130, v130, v11 :: v_dual_add_f32 v129, v129, v12
	v_dual_add_f32 v127, v127, v13 :: v_dual_add_f32 v128, v128, v14
	v_dual_add_f32 v135, v135, v15 :: v_dual_add_f32 v136, v136, v16
	v_add_f32_e32 v133, v133, v1
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v121, v134
	s_xor_b32 s4, s26, -1
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s4
	s_mov_b32 s28, -1
	v_dual_add_f32 v100, v100, v2 :: v_dual_add_f32 v69, v69, v3
	v_dual_add_f32 v70, v70, v4 :: v_dual_add_f32 v67, v67, v5
	v_add_f32_e32 v68, v68, v6
	v_dual_add_f32 v66, v66, v7 :: v_dual_add_f32 v65, v65, v8
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_115
.LBB1_118:                              ;   Parent Loop BB1_116 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_b32 s4, s29, s25
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_mov_b32 s31, s5
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[8:9], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	v_add_co_u32 v1, s33, s30, v111
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s36, s34, v109
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s36
	v_add_co_u32 v5, s34, s34, v110
	v_add_co_u32 v7, s30, s30, v120
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s34
	v_add_co_ci_u32_e64 v2, null, s31, 0, s33
	global_load_b64 v[96:97], v[3:4], off offset:40
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	v_add_nc_u32_e32 v0, s27, v121
	global_load_b64 v[98:99], v[5:6], off offset:40
	s_clause 0x1
	global_load_b64 v[112:113], v[1:2], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_b64 v[72:75], v197 offset1:32
	ds_load_2addr_b64 v[76:79], v0 offset1:32
	ds_load_2addr_b64 v[80:83], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[84:87], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[88:91], v197 offset0:192 offset1:224
	ds_load_2addr_b64 v[92:95], v0 offset0:64 offset1:96
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off offset:28 ; 4-byte Folded Reload
	s_and_b32 s27, s28, s7
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s27
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[92:93], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[92:93], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[92:93], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[92:93], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[78:79], v[74:75], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[78:79], v[82:83], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[78:79], v[86:87], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[78:79], v[90:91], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[94:95], v[74:75], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[94:95], v[82:83], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[94:95], v[86:87], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[94:95], v[90:91], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_cndmask_b32_e64 v119, 0, v97, s0
	v_cndmask_b32_e64 v118, 0, v96, s0
	v_cndmask_b32_e64 v117, 0, v99, s1
	v_cndmask_b32_e64 v116, 0, v98, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:16
	scratch_load_b32 v0, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_clause 0x6                            ; 28-byte Folded Spill
	scratch_store_b32 off, v65, off
	scratch_store_b32 off, v66, off offset:4
	scratch_store_b32 off, v68, off offset:8
	scratch_store_b32 off, v67, off offset:12
	scratch_store_b32 off, v70, off offset:16
	scratch_store_b32 off, v69, off offset:20
	scratch_store_b32 off, v100, off offset:24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_120
; %bb.119:                              ;   in Loop: Header=BB1_118 Depth=2
	scratch_load_b96 v[65:67], off, off offset:88 ; 12-byte Folded Reload
	s_add_co_i32 s4, s4, 1
	s_xor_b32 s33, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[4:5], s[14:15]
	s_add_co_i32 s4, s29, s6
	s_lshl_b32 s29, s33, 2
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[36:37], s[4:5], 0x88
	s_mulk_i32 s4, 0x88
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_lshl_b32 s30, s33, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[18:19], s[34:35]
	s_mov_b32 s31, s5
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s34, v110
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_add_nc_u64 s[36:37], s[16:17], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[36:37], s[30:31]
	s_wait_loadcnt 0x0
	v_add_co_u32 v0, vcc_lo, v65, s4
	scratch_load_b32 v65, off, off offset:104 ; 4-byte Folded Reload
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v66, vcc_lo
	v_add_co_u32 v78, vcc_lo, v0, s29
	scratch_load_b32 v0, off, off offset:100 ; 4-byte Folded Reload
	v_add_co_u32 v72, s4, s34, v109
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v79, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, s4, s30, v111
	global_load_b64 v[118:119], v[72:73], off offset:8
	v_add_co_u32 v82, s29, s30, v120
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s31, 0, s4
	v_add_co_ci_u32_e64 v83, null, s31, 0, s29
	s_wait_loadcnt 0x2
	v_mad_co_i64_i32 v[76:77], null, 0x48, v65, s[34:35]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v72, vcc_lo, v76, v0
	global_load_b64 v[116:117], v[74:75], off offset:8
	global_load_b32 v0, v[78:79], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, 0, v77, vcc_lo
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:68 ; 4-byte Folded Spill
	s_clause 0x1
	global_load_b64 v[112:113], v[80:81], off offset:8
	global_load_b64 v[114:115], v[82:83], off offset:8
	global_load_b32 v0, v[72:73], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
.LBB1_120:                              ; %.preheader468.i221
                                        ;   in Loop: Header=BB1_118 Depth=2
	scratch_load_b32 v65, off, off offset:76 ; 4-byte Folded Reload
	s_xor_b32 s4, s27, -1
	s_and_b32 s27, s26, exec_lo
	s_cselect_b32 s27, s23, 0x3c00
	v_add_nc_u32_e32 v0, 0, v121
	ds_load_2addr_b64 v[198:201], v197 offset1:32
	ds_load_2addr_b64 v[202:205], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[206:209], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[210:213], v197 offset0:192 offset1:224
	v_add_nc_u32_e32 v0, 0x2000, v0
	s_cselect_b32 s29, s13, 0x3400
	v_mov_b32_e32 v134, v121
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v74, s27, v65
	scratch_load_b32 v65, off, off offset:80 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x2000
	ds_load_2addr_b32 v[110:111], v74 offset1:1
	ds_load_2addr_b32 v[108:109], v74 offset0:2 offset1:3
	ds_load_2addr_b64 v[214:217], v0 offset1:32
	ds_load_2addr_b64 v[218:221], v0 offset0:64 offset1:96
	ds_load_2addr_b32 v[104:105], v74 offset0:4 offset1:5
	ds_load_2addr_b32 v[106:107], v74 offset0:6 offset1:7
	ds_load_2addr_b32 v[102:103], v74 offset0:8 offset1:9
	ds_load_2addr_b32 v[98:99], v74 offset0:10 offset1:11
	ds_load_2addr_b32 v[100:101], v74 offset0:12 offset1:13
	ds_load_2addr_b32 v[96:97], v74 offset0:14 offset1:15
	ds_load_2addr_b32 v[90:91], v74 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v74 offset0:34 offset1:35
	ds_load_2addr_b32 v[84:85], v74 offset0:36 offset1:37
	ds_load_2addr_b32 v[82:83], v74 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v74 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v74 offset0:42 offset1:43
	ds_load_2addr_b32 v[76:77], v74 offset0:44 offset1:45
	ds_load_2addr_b32 v[74:75], v74 offset0:46 offset1:47
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0xf
	v_wmma_i32_16x16x32_iu4 v[57:64], v[214:215], v[198:199], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[214:215], v[202:203], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[214:215], v[206:207], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[214:215], v[210:211], v[33:40] neg_lo:[0,1,0]
	s_wait_dscnt 0xe
	v_wmma_i32_16x16x32_iu4 v[25:32], v[218:219], v[198:199], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[218:219], v[202:203], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[218:219], v[206:207], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[218:219], v[210:211], v[1:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[216:217], v[200:201], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[216:217], v[204:205], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[216:217], v[208:209], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[216:217], v[212:213], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[220:221], v[200:201], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[220:221], v[204:205], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[220:221], v[208:209], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[220:221], v[212:213], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v72, s29, v65
	ds_load_2addr_b32 v[94:95], v72 offset1:1
	ds_load_2addr_b32 v[92:93], v72 offset0:32 offset1:33
	ds_load_2addr_b32 v[88:89], v72 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v72 offset0:96 offset1:97
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB1_117
; %bb.121:                              ; %.preheader469.i290
                                        ;   in Loop: Header=BB1_118 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v0, off, off offset:84
	scratch_load_b32 v65, off, off offset:68
	s_and_b32 s4, s28, exec_lo
	s_cselect_b32 s4, s13, 0x3400
	s_cselect_b32 s27, s23, 0x3c00
	v_cndmask_b32_e64 v119, 0, v119, s0
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v0, v0, v65
	scratch_load_b32 v65, off, off offset:72 ; 4-byte Folded Reload
	v_cvt_f32_f16_e32 v0, v0.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v0, 0, v0, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v120, 0, v65, s2
	scratch_load_b32 v65, off, off offset:108 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v198, s4, v65
	v_add_nc_u32_e32 v199, s27, v65
	scratch_load_b32 v65, off, off offset:28 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x1000
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v65, v[118:119], v[112:113] offset1:8
	scratch_load_b32 v65, off, off offset:32 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v65, v[116:117], v[114:115] offset1:8
	ds_store_b32 v198, v120
	ds_store_b32 v199, v0
	s_branch .LBB1_117
.LBB1_122:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:1280
	global_load_b32 v0, v[9:10], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_38
.LBB1_123:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:2560
	global_load_b32 v0, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_39
.LBB1_124:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:3840
	global_load_b32 v0, v[9:10], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_40
.LBB1_125:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:5120
	global_load_b32 v0, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_41
.LBB1_126:
	v_mad_co_i64_i32 v[9:10], null, s12, v8, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:6400
	global_load_b32 v0, v[9:10], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:256
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB1_42
.LBB1_127:
	v_mad_co_i64_i32 v[9:10], null, s12, v7, 0
	v_lshlrev_b64_e32 v[11:12], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[9:10], 2, v[9:10]
	v_add_co_u32 v0, s3, s20, v9
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, s21, v10, s3
	v_add_co_u32 v9, s3, v0, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v10, null, v10, v12, s3
	ds_load_b32 v11, v3 offset:7680
	global_load_b32 v0, v[9:10], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v11, v0
	global_store_b32 v[9:10], v0, off offset:384
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_b32 s3, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s4, s3
	s_cbranch_execnz .LBB1_43
	s_branch .LBB1_44
.LBB1_128:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:1280
	global_load_b32 v0, v[11:12], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_48
.LBB1_129:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:2560
	global_load_b32 v0, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_49
.LBB1_130:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:3840
	global_load_b32 v0, v[11:12], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:128
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_50
.LBB1_131:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:5120
	global_load_b32 v0, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_51
.LBB1_132:
	v_mad_co_i64_i32 v[11:12], null, s12, v10, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:6400
	global_load_b32 v0, v[11:12], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:256
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execz .LBB1_52
.LBB1_133:
	v_mad_co_i64_i32 v[11:12], null, s12, v9, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[11:12]
	v_add_co_u32 v0, s5, s20, v11
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, s21, v12, s5
	v_add_co_u32 v11, s5, v0, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v12, v14, s5
	ds_load_b32 v13, v3 offset:7680
	global_load_b32 v0, v[11:12], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[11:12], v0, off offset:384
	s_or_b32 exec_lo, exec_lo, s6
	s_and_b32 s5, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s6, s5
	s_cbranch_execnz .LBB1_53
	s_branch .LBB1_54
.LBB1_134:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:1280
	global_load_b32 v0, v[13:14], off
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s15, s7
	s_cbranch_execz .LBB1_58
.LBB1_135:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:2560
	global_load_b32 v0, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:128
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s15
	s_and_b32 s7, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_59
.LBB1_136:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:3840
	global_load_b32 v0, v[13:14], off offset:128
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:128
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_60
.LBB1_137:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:5120
	global_load_b32 v0, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_61
.LBB1_138:
	v_mad_co_i64_i32 v[13:14], null, s12, v12, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:6400
	global_load_b32 v0, v[13:14], off offset:256
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:256
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execz .LBB1_62
.LBB1_139:
	v_mad_co_i64_i32 v[13:14], null, s12, v11, 0
	v_lshlrev_b64_e32 v[15:16], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[13:14], 2, v[13:14]
	v_add_co_u32 v0, s7, s20, v13
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, s21, v14, s7
	v_add_co_u32 v13, s7, v0, v15
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v14, null, v14, v16, s7
	ds_load_b32 v15, v3 offset:7680
	global_load_b32 v0, v[13:14], off offset:384
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v15, v0
	global_store_b32 v[13:14], v0, off offset:384
	s_or_b32 exec_lo, exec_lo, s8
	s_and_b32 s7, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s8, s7
	s_cbranch_execnz .LBB1_63
	s_branch .LBB1_64
.LBB1_140:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3
	global_load_b32 v0, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_82
.LBB1_141:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:1280
	global_load_b32 v0, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_83
.LBB1_142:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:2560
	global_load_b32 v0, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s8, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_84
.LBB1_143:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:3840
	global_load_b32 v0, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_85
.LBB1_144:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:5120
	global_load_b32 v0, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s15, s9, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s15
	s_cbranch_execz .LBB1_86
.LBB1_145:
	v_mad_co_i64_i32 v[5:6], null, s12, v8, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	ds_load_b32 v13, v3 offset:6400
	global_load_b32 v0, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v13, v0
	global_store_b32 v[5:6], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_87
.LBB1_146:
	v_mad_co_i64_i32 v[5:6], null, s12, v7, 0
	v_lshlrev_b64_e32 v[13:14], 2, v[1:2]
	ds_load_b32 v7, v3 offset:7680
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v5, vcc_lo, v0, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, v6, v14, vcc_lo
	global_load_b32 v0, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_88
	s_branch .LBB1_89
.LBB1_147:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3
	global_load_b32 v0, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_91
.LBB1_148:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:1280
	global_load_b32 v0, v[5:6], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_92
.LBB1_149:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:2560
	global_load_b32 v0, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_93
.LBB1_150:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:3840
	global_load_b32 v0, v[5:6], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_94
.LBB1_151:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:5120
	global_load_b32 v0, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_95
.LBB1_152:
	v_mad_co_i64_i32 v[5:6], null, s12, v10, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:6400
	global_load_b32 v0, v[5:6], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_96
.LBB1_153:
	v_mad_co_i64_i32 v[5:6], null, s12, v9, 0
	v_lshlrev_b64_e32 v[7:8], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[5:6], 2, v[5:6]
	v_add_co_u32 v0, vcc_lo, s20, v5
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, s21, v6, vcc_lo
	v_add_co_u32 v5, vcc_lo, v0, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v6, null, v6, v8, vcc_lo
	ds_load_b32 v7, v3 offset:7680
	global_load_b32 v0, v[5:6], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v7, v0
	global_store_b32 v[5:6], v0, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_97
	s_branch .LBB1_98
.LBB1_154:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3
	global_load_b32 v0, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_100
.LBB1_155:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:1280
	global_load_b32 v0, v[4:5], off offset:64
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:64
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_101
.LBB1_156:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:2560
	global_load_b32 v0, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s8, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_102
.LBB1_157:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:3840
	global_load_b32 v0, v[4:5], off offset:192
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:192
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_103
.LBB1_158:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:5120
	global_load_b32 v0, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s9, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_104
.LBB1_159:
	v_mad_co_i64_i32 v[4:5], null, s12, v12, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:6400
	global_load_b32 v0, v[4:5], off offset:320
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:320
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB1_105
.LBB1_160:
	v_mad_co_i64_i32 v[4:5], null, s12, v11, 0
	v_lshlrev_b64_e32 v[6:7], 2, v[1:2]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[4:5], 2, v[4:5]
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, s21, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v5, null, v5, v7, vcc_lo
	ds_load_b32 v6, v3 offset:7680
	global_load_b32 v0, v[4:5], off offset:448
	s_wait_loadcnt_dscnt 0x0
	v_add_f32_e32 v0, v6, v0
	global_store_b32 v[4:5], v0, off offset:448
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s10, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s1
	s_cbranch_execnz .LBB1_106
	s_branch .LBB1_107
.LBB1_161:                              ; %Flow1909
	s_clause 0x3                            ; 16-byte Folded Reload
	scratch_load_b32 v17, off, off offset:112
	scratch_load_b32 v18, off, off offset:116
	scratch_load_b32 v14, off, off offset:120
	scratch_load_b32 v15, off, off offset:124
.LBB1_162:                              ; %.preheader466.i24
	s_wait_loadcnt 0x3
	v_lshrrev_b32_e32 v1, 4, v17
	s_wait_loadcnt 0x2
	v_mul_u32_u24_e32 v0, 0x500, v18
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v2, 2, v15
	v_and_b32_e32 v1, 15, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add3_u32 v3, 0, v0, v2
	v_or_b32_e32 v0, s22, v15
	v_mul_u32_u24_e32 v2, 0x50, v15
	v_or_b32_e32 v4, s11, v1
	v_lshlrev_b32_e32 v5, 2, v1
	v_mad_u32_u24 v6, 0x280, v14, v3
	v_cmp_gt_i32_e64 s7, s12, v0
	ds_store_2addr_b32 v6, v179, v186 offset1:20
	ds_store_2addr_b32 v6, v185, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v182, v183 offset0:80 offset1:100
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	v_add3_u32 v2, 0, v2, v5
	ds_store_2addr_b32 v6, v180, v181 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s0, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB1_164
; %bb.163:
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
.LBB1_164:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_166
; %bb.165:
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
.LBB1_166:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v1
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_168
; %bb.167:
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
.LBB1_168:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_170
; %bb.169:
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
.LBB1_170:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v1
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_172
; %bb.171:
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
.LBB1_172:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_174
; %bb.173:
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
.LBB1_174:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v1, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v1
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_176
; %bb.175:
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
.LBB1_176:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v14
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB1_178
; %bb.177:
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
.LBB1_178:                              ; %.preheader.1.i62
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s1, s14, v6
	ds_store_2addr_b32 v3, v176, v177 offset1:20
	ds_store_2addr_b32 v3, v174, v175 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v172, v173 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v170, v171 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB1_180
; %bb.179:
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
.LBB1_180:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
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
.LBB1_188:                              ; %.preheader.2.i73
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 32, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s3, s14, v8
	ds_store_2addr_b32 v3, v167, v168 offset1:20
	ds_store_2addr_b32 v3, v165, v166 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v163, v164 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v161, v162 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB1_190
; %bb.189:
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
.LBB1_190:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
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
.LBB1_198:                              ; %.preheader.3.i84
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v10, 48, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s5, s14, v10
	ds_store_2addr_b32 v3, v159, v160 offset1:20
	ds_store_2addr_b32 v3, v157, v158 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v155, v156 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v153, v154 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB1_200
; %bb.199:
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
.LBB1_200:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
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
.LBB1_208:                              ; %.preheader465.1.i95
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v1, 16, v0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s7, s12, v1
	ds_store_2addr_b32 v3, v152, v151 offset1:20
	ds_store_2addr_b32 v3, v149, v150 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v148, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v145, v146 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_210
; %bb.209:
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
.LBB1_210:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB1_212
; %bb.211:
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
.LBB1_212:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v1, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v1
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_214
; %bb.213:
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
.LBB1_214:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB1_216
; %bb.215:
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
.LBB1_216:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v1, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v1
	s_and_b32 s10, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_218
; %bb.217:
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
.LBB1_218:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB1_220
; %bb.219:
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
.LBB1_220:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v1, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v1
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB1_222
; %bb.221:
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
.LBB1_222:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB1_224
; %bb.223:
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
.LBB1_224:                              ; %.preheader.1.1.i108
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v143, v144 offset1:20
	ds_store_2addr_b32 v3, v141, v142 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v139, v140 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v137, v138 offset0:120 offset1:140
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
.LBB1_233:                              ; %.preheader.2.1.i117
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v132, v131 offset1:20
	ds_store_2addr_b32 v3, v130, v129 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v127, v128 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v135, v136 offset0:120 offset1:140
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
.LBB1_242:                              ; %.preheader.3.1.i126
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v133, v100 offset1:20
	ds_store_2addr_b32 v3, v69, v70 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v67, v68 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v66, v65 offset0:120 offset1:140
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
.LBB1_251:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_mov_b32 s0, -1
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
.LBB1_252:                              ; %Flow1914
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB1_254
; %bb.253:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit.sink.split
	global_inv scope:SCOPE_SE
.LBB1_254:                              ; %_ZL42gemm_mq4g256v2_residual_mmq_iu4_gfx12_bodyILb1EEvPKcPK12block_i4_128Pfiii.exit
	s_endpgm
.LBB1_255:
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
	s_cbranch_execz .LBB1_182
.LBB1_256:
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
	s_cbranch_execz .LBB1_183
.LBB1_257:
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
	s_cbranch_execz .LBB1_184
.LBB1_258:
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
	s_cbranch_execz .LBB1_185
.LBB1_259:
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
	s_cbranch_execz .LBB1_186
.LBB1_260:
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
	s_cbranch_execnz .LBB1_187
	s_branch .LBB1_188
.LBB1_261:
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
	s_cbranch_execz .LBB1_192
.LBB1_262:
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
	s_cbranch_execz .LBB1_193
.LBB1_263:
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
	s_cbranch_execz .LBB1_194
.LBB1_264:
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
	s_cbranch_execz .LBB1_195
.LBB1_265:
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
	s_cbranch_execz .LBB1_196
.LBB1_266:
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
	s_cbranch_execnz .LBB1_197
	s_branch .LBB1_198
.LBB1_267:
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
	s_cbranch_execz .LBB1_202
.LBB1_268:
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
	s_cbranch_execz .LBB1_203
.LBB1_269:
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
	s_cbranch_execz .LBB1_204
.LBB1_270:
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
	s_cbranch_execz .LBB1_205
.LBB1_271:
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
	s_cbranch_execz .LBB1_206
.LBB1_272:
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
	s_cbranch_execnz .LBB1_207
	s_branch .LBB1_208
.LBB1_273:
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
	s_cbranch_execz .LBB1_226
.LBB1_274:
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
	s_cbranch_execz .LBB1_227
.LBB1_275:
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
	s_cbranch_execz .LBB1_228
.LBB1_276:
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
	s_cbranch_execz .LBB1_229
.LBB1_277:
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
	s_cbranch_execz .LBB1_230
.LBB1_278:
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
	s_cbranch_execz .LBB1_231
.LBB1_279:
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
	s_cbranch_execnz .LBB1_232
	s_branch .LBB1_233
.LBB1_280:
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
	s_cbranch_execz .LBB1_235
.LBB1_281:
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
	s_cbranch_execz .LBB1_236
.LBB1_282:
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
	s_cbranch_execz .LBB1_237
.LBB1_283:
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
	s_cbranch_execz .LBB1_238
.LBB1_284:
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
	s_cbranch_execz .LBB1_239
.LBB1_285:
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
	s_cbranch_execz .LBB1_240
.LBB1_286:
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
	s_cbranch_execnz .LBB1_241
	s_branch .LBB1_242
.LBB1_287:
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
	s_cbranch_execz .LBB1_244
.LBB1_288:
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
	s_cbranch_execz .LBB1_245
.LBB1_289:
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
	s_cbranch_execz .LBB1_246
.LBB1_290:
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
	s_cbranch_execz .LBB1_247
.LBB1_291:
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
	s_cbranch_execz .LBB1_248
.LBB1_292:
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
	s_cbranch_execz .LBB1_249
.LBB1_293:
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
	s_cbranch_execnz .LBB1_250
	s_branch .LBB1_251
.Lfunc_end1:
	.size	gemm_mq4g256v2_residual_mmq_iu4, .Lfunc_end1-gemm_mq4g256v2_residual_mmq_iu4
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 132
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.private_seg_size, 132
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 30384
; TotalNumSgprs: 43
; NumVgprs: 256
; ScratchSize: 132
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
	s_lshl_b32 s21, ttmp9, 7
	s_lshl_b32 s20, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s21, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s20, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB2_105
; %bb.1:                                ; %.preheader476.i
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
	s_mul_i32 s8, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s20, v2
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
.LBB2_4:                                ; %.preheader471.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s21, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s8, v5, s[4:5]
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
	v_dual_mov_b32 v18, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s3, s14, -1
	v_dual_mov_b32 v26, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v17, 0 :: v_dual_add_nc_u32 v12, s20, v11
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v4, s21, v11
	v_dual_mov_b32 v22, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v5, s3, v12
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v6, 64, v4
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s3, v13
	v_lshrrev_b32_e32 v25, 5, v0
	v_dual_mov_b32 v148, 0 :: v_dual_and_b32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[126:127], null, 0x48, v5, v[3:4]
	v_min_i32_e32 v5, s2, v6
	v_mad_co_u64_u32 v[127:128], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[128:129], null, s8, v4, v[3:4]
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_mad_co_u64_u32 v[129:130], null, s8, v5, v[3:4]
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b64 v[3:4], v126, s[6:7] offset:8
	global_load_b64 v[5:6], v127, s[6:7] offset:8
	s_clause 0x1
	global_load_b64 v[7:8], v128, s[4:5] offset:8
	global_load_b64 v[9:10], v129, s[4:5] offset:8
	v_or_b32_e32 v15, 8, v25
	v_and_or_b32 v16, v25, 6, v14
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v11, 3, v11
	v_cmp_gt_i32_e64 s0, s14, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_or_b32 v14, v15, 14, v14
	v_cmp_gt_i32_e64 s1, s14, v13
	v_lshl_or_b32 v15, v16, 8, v11
	v_dual_mov_b32 v183, 0 :: v_dual_and_b32 v24, 15, v0
	v_mov_b32_e32 v153, 0
	v_lshl_or_b32 v11, v14, 8, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v12, 0, v15
	v_bfe_u32 v23, v0, 4, 1
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v155, 0
	v_add_nc_u32_e32 v11, 0, v11
	scratch_store_b32 off, v12, off offset:44 ; 4-byte Folded Spill
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v159, 0
	scratch_store_b32 off, v11, off offset:48 ; 4-byte Folded Spill
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v185, 0 :: v_dual_mov_b32 v186, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	s_wait_loadcnt 0x1
	ds_store_2addr_stride64_b64 v12, v[3:4], v[7:8] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v11, v[5:6], v[9:10] offset1:8
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_15
; %bb.6:                                ; %.preheader470.lr.ph.i
	v_and_b32_e32 v3, 31, v0
	v_lshrrev_b32_e32 v4, 6, v0
	scratch_store_b32 off, v0, off offset:96 ; 4-byte Folded Spill
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v0, 4, v1
	v_add_nc_u32_e32 v7, s21, v2
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v8, 2, v1
	v_mov_b32_e32 v183, 0
	scratch_store_b32 off, v0, off offset:68 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 2, v1
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v6, s20, v2
	v_dual_mov_b32 v185, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, 0
	scratch_store_b32 off, v24, off offset:100 ; 4-byte Folded Spill
	v_dual_mov_b32 v186, 0 :: v_dual_lshlrev_b32 v9, 6, v23
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v10, 3, v24
	scratch_store_b32 off, v0, off offset:56 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, 0
	v_min_i32_e32 v1, s2, v7
	v_dual_mov_b32 v30, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_cmp_gt_i32_e64 s2, s14, v6
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v147, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[11:12], null, s8, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	v_dual_mov_b32 v182, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v173, 0
	v_mad_co_u64_u32 v[12:13], null, s8, v1, v[12:13]
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	v_min_i32_e32 v0, s3, v6
	v_lshlrev_b32_e32 v6, 9, v5
	v_lshl_add_u32 v5, v5, 11, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v171, 0
	scratch_store_b32 off, v0, off offset:76 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v2, v8
	v_lshlrev_b32_e32 v2, 8, v4
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v143, 0
	s_clause 0x1                            ; 16-byte Folded Spill
	scratch_store_b32 off, v0, off offset:92
	scratch_store_b96 off, v[11:13], off offset:80
	v_add3_u32 v0, 0, v2, v9
	scratch_store_b32 off, v25, off offset:104 ; 4-byte Folded Spill
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v26, 0 :: v_dual_add_nc_u32 v197, v5, v3
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v18, 0
	v_cmp_gt_i32_e64 s3, s12, v7
	v_lshl_or_b32 v68, v4, 10, v3
	scratch_store_b32 off, v0, off offset:60 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v6, v10
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s27, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	scratch_store_b32 off, v0, off offset:64 ; 4-byte Folded Spill
	s_branch .LBB2_8
.LBB2_7:                                ;   in Loop: Header=BB2_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s11
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_14
.LBB2_8:                                ; %.preheader470.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_10 Depth 2
	s_add_co_i32 s24, s10, 1
	s_mov_b32 s11, s9
	s_lshl_b32 s25, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s29, 0
	s_mov_b32 s26, -1
	s_mov_b32 s28, 0
	s_cselect_b32 s11, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_branch .LBB2_10
.LBB2_9:                                ;   in Loop: Header=BB2_10 Depth=2
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_dual_mul_f32 v239, v94, v86 :: v_dual_mul_f32 v242, v94, v84
	v_dual_mul_f32 v243, v94, v82 :: v_dual_mul_f32 v246, v94, v80
	v_mul_f32_e32 v247, v94, v78
	v_mul_f32_e32 v253, v92, v90
	v_mul_f32_e32 v254, v92, v86
	v_dual_mul_f32 v168, v92, v84 :: v_dual_mul_f32 v177, v92, v82
	v_dual_mul_f32 v190, v92, v80 :: v_dual_mul_f32 v191, v92, v78
	v_cvt_f32_i32_e32 v95, v95
	v_dual_mul_f32 v241, v94, v87 :: v_dual_mul_f32 v244, v94, v85
	v_dual_mul_f32 v245, v94, v83 :: v_dual_mul_f32 v248, v94, v81
	v_mul_f32_e32 v249, v94, v79
	v_cvt_f32_i32_e32 v93, v93
	v_mul_f32_e32 v255, v92, v91
	v_dual_mul_f32 v178, v92, v87 :: v_dual_mul_f32 v189, v92, v85
	v_mul_f32_e32 v0, v92, v83
	v_dual_mul_f32 v192, v92, v81 :: v_dual_mul_f32 v65, v92, v79
	v_dual_mul_f32 v26, v239, v26 :: v_dual_mul_f32 v27, v242, v27
	v_dual_mul_f32 v28, v243, v28 :: v_dual_mul_f32 v29, v246, v29
	v_dual_mul_f32 v30, v247, v30 :: v_dual_mul_f32 v17, v253, v17
	v_dual_mul_f32 v18, v254, v18 :: v_dual_mul_f32 v19, v168, v19
	v_dual_mul_f32 v20, v177, v20 :: v_dual_mul_f32 v21, v190, v21
	v_mul_f32_e32 v22, v191, v22
	v_dual_fmac_f32 v26, v241, v95 :: v_dual_fmac_f32 v27, v244, v95
	v_dual_fmac_f32 v28, v245, v95 :: v_dual_fmac_f32 v29, v248, v95
	v_dual_fmac_f32 v30, v249, v95 :: v_dual_fmac_f32 v17, v255, v93
	v_dual_fmac_f32 v18, v178, v93 :: v_dual_fmac_f32 v19, v189, v93
	v_fmac_f32_e32 v20, v0, v93
	v_dual_fmac_f32 v21, v192, v93 :: v_dual_fmac_f32 v22, v65, v93
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_dual_add_f32 v151, v151, v26 :: v_dual_add_f32 v148, v148, v27
	v_dual_add_f32 v149, v149, v28 :: v_dual_add_f32 v146, v146, v29
	v_dual_add_f32 v147, v147, v30 :: v_dual_add_f32 v142, v142, v17
	v_dual_add_f32 v143, v143, v18 :: v_dual_add_f32 v140, v140, v19
	v_dual_add_f32 v141, v141, v20 :: v_dual_add_f32 v138, v138, v21
	v_add_f32_e32 v139, v139, v22
	s_clause 0xa                            ; 44-byte Folded Reload
	scratch_load_b32 v29, off, off offset:36 th:TH_LOAD_LU
	scratch_load_b32 v27, off, off offset:28 th:TH_LOAD_LU
	scratch_load_b32 v28, off, off offset:32 th:TH_LOAD_LU
	scratch_load_b32 v30, off, off offset:40 th:TH_LOAD_LU
	scratch_load_b32 v26, off, off offset:24 th:TH_LOAD_LU
	scratch_load_b32 v21, off, off offset:16 th:TH_LOAD_LU
	scratch_load_b32 v22, off, off offset:20 th:TH_LOAD_LU
	scratch_load_b32 v19, off, off offset:8 th:TH_LOAD_LU
	scratch_load_b32 v20, off, off offset:12 th:TH_LOAD_LU
	scratch_load_b32 v18, off, off offset:4 th:TH_LOAD_LU
	scratch_load_b32 v17, off, off th:TH_LOAD_LU
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v114, v57
	v_cvt_f32_i32_e32 v115, v58
	v_dual_mul_f32 v116, v104, v94 :: v_dual_mul_f32 v117, v106, v94
	v_cvt_f32_i32_e32 v118, v59
	v_cvt_f32_i32_e32 v119, v60
	v_mul_f32_e32 v198, v102, v94
	v_mul_f32_e32 v199, v98, v94
	v_cvt_f32_i32_e32 v200, v61
	v_cvt_f32_i32_e32 v201, v62
	v_mul_f32_e32 v202, v100, v94
	v_mul_f32_e32 v203, v96, v94
	v_cvt_f32_i32_e32 v204, v63
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v205, v110, v92 :: v_dual_mul_f32 v206, v108, v92
	v_cvt_f32_i32_e32 v207, v49
	v_cvt_f32_i32_e32 v208, v50
	v_dual_mul_f32 v209, v104, v92 :: v_dual_mul_f32 v210, v106, v92
	v_cvt_f32_i32_e32 v211, v51
	v_cvt_f32_i32_e32 v212, v52
	v_mul_f32_e32 v213, v102, v92
	v_mul_f32_e32 v214, v98, v92
	v_cvt_f32_i32_e32 v215, v53
	v_cvt_f32_i32_e32 v216, v54
	v_mul_f32_e32 v217, v100, v92
	v_cvt_f32_i32_e32 v218, v55
	v_cvt_f32_i32_e32 v56, v56
	v_dual_mul_f32 v219, v110, v88 :: v_dual_mul_f32 v220, v108, v88
	v_cvt_f32_i32_e32 v221, v41
	v_cvt_f32_i32_e32 v222, v42
	v_dual_mul_f32 v223, v104, v88 :: v_dual_mul_f32 v224, v106, v88
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v15, v15
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v227, v102, v88
	v_dual_mul_f32 v228, v98, v88 :: v_dual_mul_f32 v231, v100, v88
	v_mul_f32_e32 v110, v110, v72
	v_mul_f32_e32 v108, v108, v72
	v_mul_f32_e32 v104, v104, v72
	v_mul_f32_e32 v106, v106, v72
	v_mul_f32_e32 v102, v102, v72
	v_mul_f32_e32 v98, v98, v72
	v_mul_f32_e32 v100, v100, v72
	v_mul_f32_e32 v234, v96, v92
	v_mul_f32_e32 v236, v96, v88
	v_mul_f32_e32 v96, v96, v72
	v_mul_f32_e32 v238, v94, v90
	v_dual_mul_f32 v250, v94, v76 :: v_dual_mul_f32 v251, v94, v74
	v_dual_mul_f32 v66, v92, v76 :: v_dual_mul_f32 v193, v92, v74
	v_mul_f32_e32 v67, v88, v90
	v_mul_f32_e32 v196, v88, v86
	v_dual_mul_f32 v70, v88, v84 :: v_dual_mul_f32 v71, v88, v82
	v_mul_f32_e32 v194, v88, v80
	v_dual_mul_f32 v120, v88, v78 :: v_dual_mul_f32 v123, v88, v76
	v_mul_f32_e32 v124, v88, v74
	v_mul_f32_e32 v90, v72, v90
	v_mul_f32_e32 v86, v72, v86
	v_mul_f32_e32 v84, v72, v84
	v_mul_f32_e32 v82, v72, v82
	v_mul_f32_e32 v80, v72, v80
	v_mul_f32_e32 v78, v72, v78
	v_mul_f32_e32 v76, v72, v76
	v_mul_f32_e32 v74, v72, v74
	v_dual_mul_f32 v57, v111, v94 :: v_dual_mul_f32 v58, v109, v94
	v_dual_mul_f32 v59, v105, v94 :: v_dual_mul_f32 v60, v107, v94
	v_mul_f32_e32 v61, v103, v94
	v_dual_mul_f32 v62, v99, v94 :: v_dual_mul_f32 v63, v101, v94
	v_dual_mul_f32 v49, v111, v92 :: v_dual_mul_f32 v50, v109, v92
	v_dual_mul_f32 v51, v105, v92 :: v_dual_mul_f32 v52, v107, v92
	v_mul_f32_e32 v53, v103, v92
	v_dual_mul_f32 v54, v99, v92 :: v_dual_mul_f32 v55, v101, v92
	v_dual_mul_f32 v41, v111, v88 :: v_dual_mul_f32 v42, v109, v88
	v_dual_mul_f32 v225, v105, v88 :: v_dual_mul_f32 v226, v107, v88
	v_mul_f32_e32 v229, v103, v88
	v_mul_f32_e32 v230, v99, v88
	v_mul_f32_e32 v232, v101, v88
	v_mul_f32_e32 v111, v111, v72
	v_mul_f32_e32 v109, v109, v72
	v_mul_f32_e32 v105, v105, v72
	v_mul_f32_e32 v107, v107, v72
	v_mul_f32_e32 v103, v103, v72
	v_mul_f32_e32 v99, v99, v72
	v_mul_f32_e32 v101, v101, v72
	v_mul_f32_e32 v233, v97, v94
	v_mul_f32_e32 v235, v97, v92
	v_mul_f32_e32 v237, v97, v88
	v_dual_mul_f32 v97, v97, v72 :: v_dual_mul_f32 v240, v94, v91
	v_mul_f32_e32 v252, v94, v77
	v_dual_mul_f32 v94, v94, v75 :: v_dual_mul_f32 v195, v92, v77
	v_mul_f32_e32 v92, v92, v75
	v_cvt_f32_i32_e32 v89, v89
	v_mul_f32_e32 v68, v88, v91
	v_mul_f32_e32 v69, v88, v87
	v_dual_mul_f32 v187, v88, v85 :: v_dual_mul_f32 v188, v88, v83
	v_dual_mul_f32 v121, v88, v81 :: v_dual_mul_f32 v122, v88, v79
	v_dual_mul_f32 v125, v88, v77 :: v_dual_mul_f32 v88, v88, v75
	v_mul_f32_e32 v91, v72, v91
	v_mul_f32_e32 v87, v72, v87
	v_mul_f32_e32 v85, v72, v85
	v_mul_f32_e32 v83, v72, v83
	v_mul_f32_e32 v81, v72, v81
	v_mul_f32_e32 v79, v72, v79
	v_dual_mul_f32 v77, v72, v77 :: v_dual_mul_f32 v72, v72, v75
	v_cvt_f32_i32_e32 v73, v73
	v_dual_mul_f32 v75, v112, v114 :: v_dual_mul_f32 v112, v113, v115
	v_dual_mul_f32 v113, v116, v118 :: v_dual_mul_f32 v114, v117, v119
	v_dual_mul_f32 v115, v198, v200 :: v_dual_mul_f32 v116, v199, v201
	v_mul_f32_e32 v117, v202, v204
	v_mul_f32_e32 v64, v203, v64
	v_dual_mul_f32 v118, v205, v207 :: v_dual_mul_f32 v119, v206, v208
	v_dual_mul_f32 v198, v209, v211 :: v_dual_mul_f32 v199, v210, v212
	v_dual_mul_f32 v200, v213, v215 :: v_dual_mul_f32 v201, v214, v216
	v_mul_f32_e32 v202, v217, v218
	v_dual_mul_f32 v56, v234, v56 :: v_dual_mul_f32 v203, v219, v221
	v_dual_mul_f32 v204, v220, v222 :: v_dual_mul_f32 v43, v223, v43
	v_dual_mul_f32 v44, v224, v44 :: v_dual_mul_f32 v45, v227, v45
	v_dual_mul_f32 v46, v228, v46 :: v_dual_mul_f32 v47, v231, v47
	v_dual_mul_f32 v48, v236, v48 :: v_dual_mul_f32 v33, v110, v33
	v_mul_f32_e32 v34, v108, v34
	v_dual_mul_f32 v35, v104, v35 :: v_dual_mul_f32 v36, v106, v36
	v_mul_f32_e32 v37, v102, v37
	v_dual_mul_f32 v38, v98, v38 :: v_dual_mul_f32 v39, v100, v39
	v_dual_mul_f32 v40, v96, v40 :: v_dual_mul_f32 v25, v238, v25
	v_dual_mul_f32 v31, v250, v31 :: v_dual_mul_f32 v32, v251, v32
	v_dual_mul_f32 v23, v66, v23 :: v_dual_mul_f32 v24, v193, v24
	v_dual_mul_f32 v9, v67, v9 :: v_dual_mul_f32 v10, v196, v10
	v_dual_mul_f32 v11, v70, v11 :: v_dual_mul_f32 v12, v71, v12
	v_dual_mul_f32 v13, v194, v13 :: v_dual_mul_f32 v14, v120, v14
	v_dual_mul_f32 v15, v123, v15 :: v_dual_mul_f32 v16, v124, v16
	v_mul_f32_e32 v1, v90, v1
	v_dual_mul_f32 v2, v86, v2 :: v_dual_mul_f32 v3, v84, v3
	v_dual_mul_f32 v4, v82, v4 :: v_dual_mul_f32 v5, v80, v5
	v_dual_mul_f32 v6, v78, v6 :: v_dual_mul_f32 v7, v76, v7
	v_dual_mul_f32 v8, v74, v8 :: v_dual_fmac_f32 v75, v57, v95
	v_dual_fmac_f32 v112, v58, v95 :: v_dual_fmac_f32 v113, v59, v95
	v_dual_fmac_f32 v114, v60, v95 :: v_dual_fmac_f32 v115, v61, v95
	v_dual_fmac_f32 v116, v62, v95 :: v_dual_fmac_f32 v117, v63, v95
	v_fmac_f32_e32 v64, v233, v95
	v_dual_fmac_f32 v118, v49, v93 :: v_dual_fmac_f32 v119, v50, v93
	v_dual_fmac_f32 v198, v51, v93 :: v_dual_fmac_f32 v199, v52, v93
	v_dual_fmac_f32 v200, v53, v93 :: v_dual_fmac_f32 v201, v54, v93
	v_fmac_f32_e32 v202, v55, v93
	v_fmac_f32_e32 v56, v235, v93
	v_dual_fmac_f32 v203, v41, v89 :: v_dual_fmac_f32 v204, v42, v89
	v_dual_fmac_f32 v43, v225, v89 :: v_dual_fmac_f32 v44, v226, v89
	v_dual_fmac_f32 v45, v229, v89 :: v_dual_fmac_f32 v46, v230, v89
	v_dual_fmac_f32 v47, v232, v89 :: v_dual_fmac_f32 v48, v237, v89
	v_dual_fmac_f32 v33, v111, v73 :: v_dual_fmac_f32 v34, v109, v73
	v_dual_fmac_f32 v35, v105, v73 :: v_dual_fmac_f32 v36, v107, v73
	v_fmac_f32_e32 v37, v103, v73
	v_dual_fmac_f32 v38, v99, v73 :: v_dual_fmac_f32 v39, v101, v73
	v_dual_fmac_f32 v40, v97, v73 :: v_dual_fmac_f32 v25, v240, v95
	v_dual_fmac_f32 v31, v252, v95 :: v_dual_fmac_f32 v32, v94, v95
	v_dual_fmac_f32 v23, v195, v93 :: v_dual_fmac_f32 v24, v92, v93
	v_dual_fmac_f32 v9, v68, v89 :: v_dual_fmac_f32 v10, v69, v89
	v_dual_fmac_f32 v11, v187, v89 :: v_dual_fmac_f32 v12, v188, v89
	v_dual_fmac_f32 v13, v121, v89 :: v_dual_fmac_f32 v14, v122, v89
	v_dual_fmac_f32 v15, v125, v89 :: v_dual_fmac_f32 v16, v88, v89
	v_fmac_f32_e32 v1, v91, v73
	v_dual_fmac_f32 v2, v87, v73 :: v_dual_fmac_f32 v3, v85, v73
	v_dual_fmac_f32 v4, v83, v73 :: v_dual_fmac_f32 v5, v81, v73
	v_dual_fmac_f32 v6, v79, v73 :: v_dual_fmac_f32 v7, v77, v73
	v_dual_fmac_f32 v8, v72, v73 :: v_dual_add_f32 v179, v179, v75
	v_dual_add_f32 v186, v186, v112 :: v_dual_add_f32 v185, v185, v113
	v_add_f32_e32 v184, v184, v114
	v_dual_add_f32 v182, v182, v115 :: v_dual_add_f32 v183, v183, v116
	v_dual_add_f32 v180, v180, v117 :: v_dual_add_f32 v181, v181, v64
	v_dual_add_f32 v175, v175, v118 :: v_dual_add_f32 v176, v176, v119
	v_dual_add_f32 v173, v173, v198 :: v_dual_add_f32 v174, v174, v199
	v_dual_add_f32 v171, v171, v200 :: v_dual_add_f32 v172, v172, v201
	v_dual_add_f32 v169, v169, v202 :: v_dual_add_f32 v170, v170, v56
	v_dual_add_f32 v166, v166, v203 :: v_dual_add_f32 v167, v167, v204
	v_dual_add_f32 v164, v164, v43 :: v_dual_add_f32 v165, v165, v44
	v_dual_add_f32 v162, v162, v45 :: v_dual_add_f32 v163, v163, v46
	v_dual_add_f32 v160, v160, v47 :: v_dual_add_f32 v161, v161, v48
	v_dual_add_f32 v158, v158, v33 :: v_dual_add_f32 v159, v159, v34
	v_dual_add_f32 v156, v156, v35 :: v_dual_add_f32 v157, v157, v36
	v_dual_add_f32 v154, v154, v37 :: v_dual_add_f32 v155, v155, v38
	v_dual_add_f32 v152, v152, v39 :: v_dual_add_f32 v153, v153, v40
	v_add_f32_e32 v150, v150, v25
	v_dual_add_f32 v144, v144, v31 :: v_dual_add_f32 v145, v145, v32
	v_dual_add_f32 v136, v136, v23 :: v_dual_add_f32 v137, v137, v24
	v_dual_add_f32 v134, v134, v9 :: v_dual_add_f32 v135, v135, v10
	v_dual_add_f32 v132, v132, v11 :: v_dual_add_f32 v133, v133, v12
	v_add_f32_e32 v131, v131, v14
	s_wait_loadcnt 0xa
	v_add_f32_e32 v29, v29, v13
	s_wait_loadcnt 0x8
	v_dual_add_f32 v27, v27, v15 :: v_dual_add_f32 v28, v28, v16
	s_wait_loadcnt 0x7
	v_add_f32_e32 v30, v30, v1
	s_wait_loadcnt 0x5
	v_dual_add_f32 v26, v26, v2 :: v_dual_add_f32 v21, v21, v3
	s_wait_loadcnt 0x3
	v_dual_add_f32 v22, v22, v4 :: v_dual_add_f32 v19, v19, v5
	s_wait_loadcnt 0x2
	v_add_f32_e32 v20, v20, v6
	s_wait_loadcnt 0x0
	v_dual_add_f32 v18, v18, v7 :: v_dual_add_f32 v17, v17, v8
	v_mov_b32_e32 v68, v130
	s_xor_b32 s8, s26, -1
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_7
.LBB2_10:                               ;   Parent Loop BB2_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s29, s25
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_mov_b32 s31, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	v_add_co_u32 v1, s33, s30, v128
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s36, s34, v126
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s36
	v_add_co_ci_u32_e64 v2, null, s31, 0, s33
	v_add_co_u32 v5, s34, s34, v127
	v_add_co_u32 v7, s30, s30, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s34
	global_load_b64 v[96:97], v[3:4], off offset:40
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	global_load_b64 v[112:113], v[1:2], off offset:40
	v_add_nc_u32_e32 v1, s27, v68
	s_clause 0xa                            ; 44-byte Folded Spill
	scratch_store_b32 off, v30, off offset:40
	scratch_store_b32 off, v29, off offset:36
	scratch_store_b32 off, v28, off offset:32
	scratch_store_b32 off, v27, off offset:28
	scratch_store_b32 off, v26, off offset:24
	scratch_store_b32 off, v22, off offset:20
	scratch_store_b32 off, v21, off offset:16
	scratch_store_b32 off, v20, off offset:12
	scratch_store_b32 off, v19, off offset:8
	scratch_store_b32 off, v18, off offset:4
	scratch_store_b32 off, v17, off
	global_load_b64 v[98:99], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_b64 v[72:75], v197 offset1:32
	ds_load_2addr_b64 v[76:79], v1 offset1:32
	ds_load_2addr_b64 v[80:83], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[84:87], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[88:91], v197 offset0:192 offset1:224
	ds_load_2addr_b64 v[92:95], v1 offset0:64 offset1:96
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off offset:44 ; 4-byte Folded Reload
	s_and_b32 s27, s28, s11
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s27
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[92:93], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[92:93], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[92:93], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[92:93], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[78:79], v[74:75], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[78:79], v[82:83], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[78:79], v[86:87], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[78:79], v[90:91], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[94:95], v[74:75], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[94:95], v[82:83], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[94:95], v[86:87], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[94:95], v[90:91], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_cndmask_b32_e64 v119, 0, v97, s0
	v_cndmask_b32_e64 v118, 0, v96, s0
	v_cndmask_b32_e64 v117, 0, v99, s1
	v_cndmask_b32_e64 v116, 0, v98, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:16
	scratch_load_b32 v0, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_10 Depth=2
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b32 v0, off, off offset:76
	scratch_load_b96 v[65:67], off, off offset:80
	s_add_co_i32 s8, s8, 1
	s_xor_b32 s33, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s29, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_mulk_i32 s8, 0x88
	s_lshl_b32 s29, s33, 2
	s_lshl_b32 s30, s33, 6
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s34, v127
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_mov_b32 s31, s9
	s_add_nc_u64 s[36:37], s[4:5], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[36:37], s[30:31]
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[76:77], null, 0x48, v0, s[34:35]
	scratch_load_b32 v0, off, off offset:72 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_add_co_u32 v78, vcc_lo, v65, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v66, vcc_lo
	v_add_co_u32 v72, s8, s34, v126
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v78, vcc_lo, v78, s29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v79, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, s8, s30, v128
	global_load_b64 v[118:119], v[72:73], off offset:8
	v_add_co_u32 v82, s29, s30, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s31, 0, s8
	v_add_co_ci_u32_e64 v83, null, s31, 0, s29
	s_wait_loadcnt 0x1
	v_add_co_u32 v72, vcc_lo, v76, v0
	global_load_b64 v[116:117], v[74:75], off offset:8
	global_load_b32 v0, v[78:79], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, 0, v77, vcc_lo
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	s_clause 0x1
	global_load_b64 v[112:113], v[80:81], off offset:8
	global_load_b64 v[114:115], v[82:83], off offset:8
	global_load_b32 v0, v[72:73], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:56 ; 4-byte Folded Spill
.LBB2_12:                               ; %.preheader468.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	scratch_load_b32 v0, off, off offset:60 ; 4-byte Folded Reload
	s_xor_b32 s8, s27, -1
	s_and_b32 s27, s26, exec_lo
	s_cselect_b32 s27, s23, 0x3c00
	v_add_nc_u32_e32 v72, 0, v68
	ds_load_2addr_b64 v[198:201], v197 offset1:32
	ds_load_2addr_b64 v[202:205], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[206:209], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[210:213], v197 offset0:192 offset1:224
	v_add_nc_u32_e32 v72, 0x2000, v72
	s_cselect_b32 s29, s13, 0x3400
	v_mov_b32_e32 v130, v68
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v74, s27, v0
	scratch_load_b32 v0, off, off offset:64 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x2000
	ds_load_2addr_b32 v[110:111], v74 offset1:1
	ds_load_2addr_b32 v[108:109], v74 offset0:2 offset1:3
	ds_load_2addr_b64 v[214:217], v72 offset1:32
	ds_load_2addr_b64 v[218:221], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[104:105], v74 offset0:4 offset1:5
	ds_load_2addr_b32 v[106:107], v74 offset0:6 offset1:7
	ds_load_2addr_b32 v[102:103], v74 offset0:8 offset1:9
	ds_load_2addr_b32 v[98:99], v74 offset0:10 offset1:11
	ds_load_2addr_b32 v[100:101], v74 offset0:12 offset1:13
	ds_load_2addr_b32 v[96:97], v74 offset0:14 offset1:15
	ds_load_2addr_b32 v[90:91], v74 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v74 offset0:34 offset1:35
	ds_load_2addr_b32 v[84:85], v74 offset0:36 offset1:37
	ds_load_2addr_b32 v[82:83], v74 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v74 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v74 offset0:42 offset1:43
	ds_load_2addr_b32 v[76:77], v74 offset0:44 offset1:45
	ds_load_2addr_b32 v[74:75], v74 offset0:46 offset1:47
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0xf
	v_wmma_i32_16x16x32_iu4 v[57:64], v[214:215], v[198:199], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[214:215], v[202:203], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[214:215], v[206:207], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[214:215], v[210:211], v[33:40] neg_lo:[0,1,0]
	s_wait_dscnt 0xe
	v_wmma_i32_16x16x32_iu4 v[25:32], v[218:219], v[198:199], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[218:219], v[202:203], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[218:219], v[206:207], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[218:219], v[210:211], v[1:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[216:217], v[200:201], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[216:217], v[204:205], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[216:217], v[208:209], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[216:217], v[212:213], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[220:221], v[200:201], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[220:221], v[204:205], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[220:221], v[208:209], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[220:221], v[212:213], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v73, s29, v0
	ds_load_2addr_b32 v[94:95], v73 offset1:1
	ds_load_2addr_b32 v[92:93], v73 offset0:32 offset1:33
	ds_load_2addr_b32 v[88:89], v73 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v73 offset0:96 offset1:97
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_9
; %bb.13:                               ; %.preheader469.i
                                        ;   in Loop: Header=BB2_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v0, off, off offset:68
	scratch_load_b32 v65, off, off offset:52
	s_and_b32 s8, s28, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	s_cselect_b32 s27, s23, 0x3c00
	v_cndmask_b32_e64 v119, 0, v119, s0
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v0, v65
	scratch_load_b32 v0, off, off offset:56 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v198, v198.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v199, 0, v0, s2
	scratch_load_b32 v0, off, off offset:92 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v0
	v_add_nc_u32_e32 v201, s27, v0
	scratch_load_b32 v0, off, off offset:44 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x1000
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:8
	scratch_load_b32 v0, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB2_9
.LBB2_14:                               ; %Flow
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v0, off, off offset:96
	scratch_load_b32 v24, off, off offset:100
	scratch_load_b32 v25, off, off offset:104
	s_wait_loadcnt 0x2
	v_bfe_u32 v23, v0, 4, 1
.LBB2_15:                               ; %.preheader466.i
	v_lshrrev_b32_e32 v0, 4, v0
	s_wait_loadcnt 0x0
	v_mul_u32_u24_e32 v1, 0x500, v25
	v_lshlrev_b32_e32 v2, 2, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_b32_e32 v4, 15, v0
	v_or_b32_e32 v0, s21, v24
	v_add3_u32 v3, 0, v1, v2
	v_mul_u32_u24_e32 v2, 0x50, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v5, 2, v4
	v_or_b32_e32 v4, s20, v4
	v_cmp_gt_i32_e64 s7, s12, v0
	v_mad_u32_u24 v6, 0x280, v23, v3
	v_ashrrev_i32_e32 v1, 31, v0
	v_add3_u32 v2, 0, v2, v5
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	ds_store_2addr_b32 v6, v179, v186 offset1:20
	ds_store_2addr_b32 v6, v185, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v182, v183 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v180, v181 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB2_17
; %bb.16:
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
.LBB2_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_19
; %bb.18:
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
.LBB2_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v6
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_21
; %bb.20:
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
.LBB2_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_23
; %bb.22:
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
.LBB2_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v6
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_25
; %bb.24:
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
.LBB2_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_27
; %bb.26:
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
.LBB2_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v6
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_29
; %bb.28:
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
.LBB2_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v23
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB2_31
; %bb.30:
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
.LBB2_31:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s1, s14, v6
	ds_store_2addr_b32 v3, v175, v176 offset1:20
	ds_store_2addr_b32 v3, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v171, v172 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v169, v170 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB2_33
; %bb.32:
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
.LBB2_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
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
.LBB2_41:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 32, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s3, s14, v8
	ds_store_2addr_b32 v3, v166, v167 offset1:20
	ds_store_2addr_b32 v3, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB2_43
; %bb.42:
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
.LBB2_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
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
.LBB2_51:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v10, 48, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s5, s14, v10
	ds_store_2addr_b32 v3, v158, v159 offset1:20
	ds_store_2addr_b32 v3, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB2_53
; %bb.52:
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
.LBB2_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
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
.LBB2_61:                               ; %.preheader465.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v12, 16, v0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s7, s12, v12
	ds_store_2addr_b32 v3, v150, v151 offset1:20
	ds_store_2addr_b32 v3, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v146, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_63
; %bb.62:
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
.LBB2_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB2_65
; %bb.64:
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
.LBB2_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v12, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v12
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_67
; %bb.66:
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
.LBB2_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB2_69
; %bb.68:
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
.LBB2_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v12, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v12
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_71
; %bb.70:
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
.LBB2_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB2_73
; %bb.72:
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
.LBB2_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v12, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v12
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB2_75
; %bb.74:
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
.LBB2_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB2_77
; %bb.76:
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
.LBB2_77:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_wait_loadcnt 0x0
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
.LBB2_86:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v134, v135 offset1:20
	ds_store_2addr_b32 v3, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v29, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v27, v28 offset0:120 offset1:140
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
.LBB2_95:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v30, v26 offset1:20
	ds_store_2addr_b32 v3, v21, v22 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v19, v20 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v18, v17 offset0:120 offset1:140
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
	s_cbranch_execz .LBB2_35
.LBB2_107:
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
	s_cbranch_execz .LBB2_36
.LBB2_108:
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
	s_cbranch_execz .LBB2_37
.LBB2_109:
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
	s_cbranch_execz .LBB2_38
.LBB2_110:
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
	s_cbranch_execz .LBB2_39
.LBB2_111:
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
	s_cbranch_execnz .LBB2_40
	s_branch .LBB2_41
.LBB2_112:
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
	s_cbranch_execz .LBB2_45
.LBB2_113:
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
	s_cbranch_execz .LBB2_46
.LBB2_114:
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
	s_cbranch_execz .LBB2_47
.LBB2_115:
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
	s_cbranch_execz .LBB2_48
.LBB2_116:
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
	s_cbranch_execz .LBB2_49
.LBB2_117:
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
	s_cbranch_execnz .LBB2_50
	s_branch .LBB2_51
.LBB2_118:
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
	s_cbranch_execz .LBB2_55
.LBB2_119:
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
	s_cbranch_execz .LBB2_56
.LBB2_120:
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
	s_cbranch_execz .LBB2_57
.LBB2_121:
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
	s_cbranch_execz .LBB2_58
.LBB2_122:
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
	s_cbranch_execz .LBB2_59
.LBB2_123:
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
	s_cbranch_execnz .LBB2_60
	s_branch .LBB2_61
.LBB2_124:
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
	s_cbranch_execz .LBB2_79
.LBB2_125:
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
	s_cbranch_execz .LBB2_80
.LBB2_126:
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
	s_cbranch_execz .LBB2_81
.LBB2_127:
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
	s_cbranch_execz .LBB2_82
.LBB2_128:
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
	s_cbranch_execz .LBB2_83
.LBB2_129:
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
	s_cbranch_execz .LBB2_84
.LBB2_130:
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
	s_cbranch_execnz .LBB2_85
	s_branch .LBB2_86
.LBB2_131:
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
	s_cbranch_execz .LBB2_88
.LBB2_132:
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
	s_cbranch_execz .LBB2_89
.LBB2_133:
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
	s_cbranch_execz .LBB2_90
.LBB2_134:
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
	s_cbranch_execz .LBB2_91
.LBB2_135:
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
	s_cbranch_execz .LBB2_92
.LBB2_136:
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
	s_cbranch_execz .LBB2_93
.LBB2_137:
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
	s_cbranch_execnz .LBB2_94
	s_branch .LBB2_95
.LBB2_138:
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
	s_cbranch_execz .LBB2_97
.LBB2_139:
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
	s_cbranch_execz .LBB2_98
.LBB2_140:
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
	s_cbranch_execz .LBB2_99
.LBB2_141:
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
	s_cbranch_execz .LBB2_100
.LBB2_142:
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
	s_cbranch_execz .LBB2_101
.LBB2_143:
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
	s_cbranch_execz .LBB2_102
.LBB2_144:
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
	s_cbranch_execnz .LBB2_103
	s_branch .LBB2_104
.Lfunc_end2:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_add, .Lfunc_end2-gemm_mq4g256v2_residual_mmq_iu4_full_add
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 112
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.private_seg_size, 112
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15560
; TotalNumSgprs: 40
; NumVgprs: 256
; ScratchSize: 112
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
	s_lshl_b32 s21, ttmp9, 7
	s_lshl_b32 s20, ttmp7, 7
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s21, s12
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s20, s14
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_105
; %bb.1:                                ; %.preheader476.i
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
	s_mul_i32 s8, s22, 0x88
	v_cmpx_gt_u32_e32 0x100, v0
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.lr.ph.i
	v_or_b32_e32 v5, s20, v2
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
.LBB3_4:                                ; %.preheader471.loopexit.i
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v7, s21, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_min_i32_e32 v5, s2, v7
	v_cmp_gt_i32_e32 vcc_lo, s12, v7
	s_wait_kmcnt 0x0
	v_mad_co_i64_i32 v[5:6], null, s8, v5, s[4:5]
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
	v_dual_mov_b32 v18, 0 :: v_dual_and_b32 v3, 3, v0
	s_add_co_i32 s3, s14, -1
	v_dual_mov_b32 v26, 0 :: v_dual_lshlrev_b32 v15, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v17, 0 :: v_dual_add_nc_u32 v12, s20, v11
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v4, s21, v11
	v_dual_mov_b32 v22, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v13, 64, v12
	s_wait_alu depctr_sa_sdst(0)
	v_min_i32_e32 v5, s3, v12
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v6, 64, v4
	v_min_i32_e32 v4, s2, v4
	v_min_i32_e32 v7, s3, v13
	v_lshrrev_b32_e32 v25, 5, v0
	v_dual_mov_b32 v148, 0 :: v_dual_and_b32 v15, 16, v15
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[126:127], null, 0x48, v5, v[3:4]
	v_min_i32_e32 v5, s2, v6
	v_mad_co_u64_u32 v[127:128], null, 0x48, v7, v[3:4]
	v_mad_co_u64_u32 v[128:129], null, s8, v4, v[3:4]
	v_bfe_u32 v14, v0, 1, 1
	v_and_or_b32 v11, v11, 15, v15
	v_mad_co_u64_u32 v[129:130], null, s8, v5, v[3:4]
	s_wait_kmcnt 0x0
	s_clause 0x1
	global_load_b64 v[3:4], v126, s[6:7] offset:8
	global_load_b64 v[5:6], v127, s[6:7] offset:8
	s_clause 0x1
	global_load_b64 v[7:8], v128, s[4:5] offset:8
	global_load_b64 v[9:10], v129, s[4:5] offset:8
	v_or_b32_e32 v15, 8, v25
	v_and_or_b32 v16, v25, 6, v14
	v_dual_mov_b32 v184, 0 :: v_dual_lshlrev_b32 v11, 3, v11
	v_cmp_gt_i32_e64 s0, s14, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_or_b32 v14, v15, 14, v14
	v_cmp_gt_i32_e64 s1, s14, v13
	v_lshl_or_b32 v15, v16, 8, v11
	v_dual_mov_b32 v183, 0 :: v_dual_and_b32 v24, 15, v0
	v_mov_b32_e32 v153, 0
	v_lshl_or_b32 v11, v14, 8, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v12, 0, v15
	v_bfe_u32 v23, v0, 4, 1
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v155, 0
	v_add_nc_u32_e32 v11, 0, v11
	scratch_store_b32 off, v12, off offset:44 ; 4-byte Folded Spill
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v159, 0
	scratch_store_b32 off, v11, off offset:48 ; 4-byte Folded Spill
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v134, 0
	v_dual_mov_b32 v161, 0 :: v_dual_mov_b32 v160, 0
	v_dual_mov_b32 v163, 0 :: v_dual_mov_b32 v162, 0
	v_dual_mov_b32 v165, 0 :: v_dual_mov_b32 v164, 0
	v_dual_mov_b32 v167, 0 :: v_dual_mov_b32 v166, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v140, 0
	v_dual_mov_b32 v143, 0 :: v_dual_mov_b32 v142, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v171, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v173, 0
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v145, 0 :: v_dual_mov_b32 v144, 0
	v_dual_mov_b32 v147, 0 :: v_dual_mov_b32 v146, 0
	v_dual_mov_b32 v149, 0 :: v_dual_mov_b32 v150, 0
	v_dual_mov_b32 v151, 0 :: v_dual_mov_b32 v180, 0
	v_dual_mov_b32 v181, 0 :: v_dual_mov_b32 v182, 0
	v_dual_mov_b32 v185, 0 :: v_dual_mov_b32 v186, 0
	v_mov_b32_e32 v179, 0
	s_mov_b32 s9, 0
	s_cmp_lt_i32 s13, 0x100
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v4, 0, v4, s0
	v_cndmask_b32_e64 v3, 0, v3, s0
	s_wait_loadcnt 0x2
	v_cndmask_b32_e64 v6, 0, v6, s1
	v_cndmask_b32_e64 v5, 0, v5, s1
	s_wait_loadcnt 0x1
	ds_store_2addr_stride64_b64 v12, v[3:4], v[7:8] offset1:8
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v11, v[5:6], v[9:10] offset1:8
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_15
; %bb.6:                                ; %.preheader470.lr.ph.i
	v_and_b32_e32 v3, 31, v0
	v_lshrrev_b32_e32 v4, 6, v0
	scratch_store_b32 off, v0, off offset:96 ; 4-byte Folded Spill
	v_bfe_u32 v5, v0, 5, 1
	v_dual_mov_b32 v151, 0 :: v_dual_lshlrev_b32 v0, 4, v1
	v_add_nc_u32_e32 v7, s21, v2
	v_dual_mov_b32 v149, 0 :: v_dual_lshlrev_b32 v8, 2, v1
	v_mov_b32_e32 v183, 0
	scratch_store_b32 off, v0, off offset:68 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v0, 2, v1
	v_dual_mov_b32 v179, 0 :: v_dual_add_nc_u32 v6, s20, v2
	v_dual_mov_b32 v185, 0 :: v_dual_lshlrev_b32 v2, 3, v2
	scratch_store_b32 off, v0, off offset:72 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, 0
	scratch_store_b32 off, v24, off offset:100 ; 4-byte Folded Spill
	v_dual_mov_b32 v186, 0 :: v_dual_lshlrev_b32 v9, 6, v23
	v_dual_mov_b32 v181, 0 :: v_dual_lshlrev_b32 v10, 3, v24
	scratch_store_b32 off, v0, off offset:56 ; 4-byte Folded Spill
	v_mov_b32_e32 v0, 0
	v_min_i32_e32 v1, s2, v7
	v_dual_mov_b32 v30, 0 :: v_dual_lshlrev_b32 v3, 3, v3
	v_cmp_gt_i32_e64 s2, s14, v6
	v_dual_mov_b32 v184, 0 :: v_dual_mov_b32 v147, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[11:12], null, s8, v1, s[4:5]
	v_ashrrev_i32_e32 v1, 31, v1
	v_dual_mov_b32 v182, 0 :: v_dual_mov_b32 v145, 0
	v_dual_mov_b32 v180, 0 :: v_dual_mov_b32 v175, 0
	v_dual_mov_b32 v150, 0 :: v_dual_mov_b32 v173, 0
	v_mad_co_u64_u32 v[12:13], null, s8, v1, v[12:13]
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	v_min_i32_e32 v0, s3, v6
	v_lshlrev_b32_e32 v6, 9, v5
	v_lshl_add_u32 v5, v5, 11, 0
	v_dual_mov_b32 v148, 0 :: v_dual_mov_b32 v171, 0
	scratch_store_b32 off, v0, off offset:76 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v2, v8
	v_lshlrev_b32_e32 v2, 8, v4
	v_dual_mov_b32 v146, 0 :: v_dual_mov_b32 v169, 0
	v_dual_mov_b32 v144, 0 :: v_dual_mov_b32 v143, 0
	s_clause 0x1                            ; 16-byte Folded Spill
	scratch_store_b32 off, v0, off offset:92
	scratch_store_b96 off, v[11:13], off offset:80
	v_add3_u32 v0, 0, v2, v9
	scratch_store_b32 off, v25, off offset:104 ; 4-byte Folded Spill
	v_dual_mov_b32 v176, 0 :: v_dual_mov_b32 v141, 0
	v_dual_mov_b32 v174, 0 :: v_dual_mov_b32 v139, 0
	v_dual_mov_b32 v172, 0 :: v_dual_mov_b32 v137, 0
	v_dual_mov_b32 v170, 0 :: v_dual_mov_b32 v167, 0
	v_dual_mov_b32 v142, 0 :: v_dual_mov_b32 v165, 0
	v_dual_mov_b32 v140, 0 :: v_dual_mov_b32 v163, 0
	v_dual_mov_b32 v138, 0 :: v_dual_mov_b32 v161, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	v_dual_mov_b32 v166, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v164, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v162, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v160, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v159, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v157, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v153, 0
	v_dual_mov_b32 v156, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v152, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v26, 0 :: v_dual_add_nc_u32 v197, v5, v3
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v18, 0
	v_cmp_gt_i32_e64 s3, s12, v7
	v_lshl_or_b32 v68, v4, 10, v3
	scratch_store_b32 off, v0, off offset:60 ; 4-byte Folded Spill
	v_add3_u32 v0, 0, v6, v10
	s_ashr_i32 s15, s14, 31
	s_movk_i32 s27, 0x1000
	s_movk_i32 s13, 0x3000
	s_movk_i32 s23, 0x3800
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s10, s9
	scratch_store_b32 off, v0, off offset:64 ; 4-byte Folded Spill
	s_branch .LBB3_8
.LBB3_7:                                ;   in Loop: Header=BB3_8 Depth=1
	s_and_b32 vcc_lo, exec_lo, s11
	s_mov_b32 s10, s24
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_14
.LBB3_8:                                ; %.preheader470.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB3_10 Depth 2
	s_add_co_i32 s24, s10, 1
	s_mov_b32 s11, s9
	s_lshl_b32 s25, s10, 1
	s_mul_u64 s[18:19], s[10:11], 0x88
	s_cmp_eq_u32 s24, s22
	s_mov_b32 s29, 0
	s_mov_b32 s26, -1
	s_mov_b32 s28, 0
	s_cselect_b32 s11, -1, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	s_branch .LBB3_10
.LBB3_9:                                ;   in Loop: Header=BB3_10 Depth=2
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v29, v29
	v_cvt_f32_i32_e32 v30, v30
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v20, v20
	v_cvt_f32_i32_e32 v21, v21
	v_cvt_f32_i32_e32 v22, v22
	v_dual_mul_f32 v239, v94, v86 :: v_dual_mul_f32 v242, v94, v84
	v_dual_mul_f32 v243, v94, v82 :: v_dual_mul_f32 v246, v94, v80
	v_mul_f32_e32 v247, v94, v78
	v_mul_f32_e32 v253, v92, v90
	v_mul_f32_e32 v254, v92, v86
	v_dual_mul_f32 v168, v92, v84 :: v_dual_mul_f32 v177, v92, v82
	v_dual_mul_f32 v190, v92, v80 :: v_dual_mul_f32 v191, v92, v78
	v_cvt_f32_i32_e32 v95, v95
	v_dual_mul_f32 v241, v94, v87 :: v_dual_mul_f32 v244, v94, v85
	v_dual_mul_f32 v245, v94, v83 :: v_dual_mul_f32 v248, v94, v81
	v_mul_f32_e32 v249, v94, v79
	v_cvt_f32_i32_e32 v93, v93
	v_mul_f32_e32 v255, v92, v91
	v_dual_mul_f32 v178, v92, v87 :: v_dual_mul_f32 v189, v92, v85
	v_mul_f32_e32 v0, v92, v83
	v_dual_mul_f32 v192, v92, v81 :: v_dual_mul_f32 v65, v92, v79
	v_dual_mul_f32 v26, v239, v26 :: v_dual_mul_f32 v27, v242, v27
	v_dual_mul_f32 v28, v243, v28 :: v_dual_mul_f32 v29, v246, v29
	v_dual_mul_f32 v30, v247, v30 :: v_dual_mul_f32 v17, v253, v17
	v_dual_mul_f32 v18, v254, v18 :: v_dual_mul_f32 v19, v168, v19
	v_dual_mul_f32 v20, v177, v20 :: v_dual_mul_f32 v21, v190, v21
	v_mul_f32_e32 v22, v191, v22
	v_dual_fmac_f32 v26, v241, v95 :: v_dual_fmac_f32 v27, v244, v95
	v_dual_fmac_f32 v28, v245, v95 :: v_dual_fmac_f32 v29, v248, v95
	v_dual_fmac_f32 v30, v249, v95 :: v_dual_fmac_f32 v17, v255, v93
	v_dual_fmac_f32 v18, v178, v93 :: v_dual_fmac_f32 v19, v189, v93
	v_fmac_f32_e32 v20, v0, v93
	v_dual_fmac_f32 v21, v192, v93 :: v_dual_fmac_f32 v22, v65, v93
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_dual_add_f32 v151, v151, v26 :: v_dual_add_f32 v148, v148, v27
	v_dual_add_f32 v149, v149, v28 :: v_dual_add_f32 v146, v146, v29
	v_dual_add_f32 v147, v147, v30 :: v_dual_add_f32 v142, v142, v17
	v_dual_add_f32 v143, v143, v18 :: v_dual_add_f32 v140, v140, v19
	v_dual_add_f32 v141, v141, v20 :: v_dual_add_f32 v138, v138, v21
	v_add_f32_e32 v139, v139, v22
	s_clause 0xa                            ; 44-byte Folded Reload
	scratch_load_b32 v29, off, off offset:36 th:TH_LOAD_LU
	scratch_load_b32 v27, off, off offset:28 th:TH_LOAD_LU
	scratch_load_b32 v28, off, off offset:32 th:TH_LOAD_LU
	scratch_load_b32 v30, off, off offset:40 th:TH_LOAD_LU
	scratch_load_b32 v26, off, off offset:24 th:TH_LOAD_LU
	scratch_load_b32 v21, off, off offset:16 th:TH_LOAD_LU
	scratch_load_b32 v22, off, off offset:20 th:TH_LOAD_LU
	scratch_load_b32 v19, off, off offset:8 th:TH_LOAD_LU
	scratch_load_b32 v20, off, off offset:12 th:TH_LOAD_LU
	scratch_load_b32 v18, off, off offset:4 th:TH_LOAD_LU
	scratch_load_b32 v17, off, off th:TH_LOAD_LU
	v_dual_mul_f32 v112, v110, v94 :: v_dual_mul_f32 v113, v108, v94
	v_cvt_f32_i32_e32 v114, v57
	v_cvt_f32_i32_e32 v115, v58
	v_dual_mul_f32 v116, v104, v94 :: v_dual_mul_f32 v117, v106, v94
	v_cvt_f32_i32_e32 v118, v59
	v_cvt_f32_i32_e32 v119, v60
	v_mul_f32_e32 v198, v102, v94
	v_mul_f32_e32 v199, v98, v94
	v_cvt_f32_i32_e32 v200, v61
	v_cvt_f32_i32_e32 v201, v62
	v_mul_f32_e32 v202, v100, v94
	v_mul_f32_e32 v203, v96, v94
	v_cvt_f32_i32_e32 v204, v63
	v_cvt_f32_i32_e32 v64, v64
	v_dual_mul_f32 v205, v110, v92 :: v_dual_mul_f32 v206, v108, v92
	v_cvt_f32_i32_e32 v207, v49
	v_cvt_f32_i32_e32 v208, v50
	v_dual_mul_f32 v209, v104, v92 :: v_dual_mul_f32 v210, v106, v92
	v_cvt_f32_i32_e32 v211, v51
	v_cvt_f32_i32_e32 v212, v52
	v_mul_f32_e32 v213, v102, v92
	v_mul_f32_e32 v214, v98, v92
	v_cvt_f32_i32_e32 v215, v53
	v_cvt_f32_i32_e32 v216, v54
	v_mul_f32_e32 v217, v100, v92
	v_cvt_f32_i32_e32 v218, v55
	v_cvt_f32_i32_e32 v56, v56
	v_dual_mul_f32 v219, v110, v88 :: v_dual_mul_f32 v220, v108, v88
	v_cvt_f32_i32_e32 v221, v41
	v_cvt_f32_i32_e32 v222, v42
	v_dual_mul_f32 v223, v104, v88 :: v_dual_mul_f32 v224, v106, v88
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v45, v45
	v_cvt_f32_i32_e32 v46, v46
	v_cvt_f32_i32_e32 v47, v47
	v_cvt_f32_i32_e32 v48, v48
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	v_cvt_f32_i32_e32 v37, v37
	v_cvt_f32_i32_e32 v38, v38
	v_cvt_f32_i32_e32 v39, v39
	v_cvt_f32_i32_e32 v40, v40
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v31, v31
	v_cvt_f32_i32_e32 v32, v32
	v_cvt_f32_i32_e32 v23, v23
	v_cvt_f32_i32_e32 v24, v24
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v13, v13
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v15, v15
	v_cvt_f32_i32_e32 v16, v16
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	v_cvt_f32_i32_e32 v6, v6
	v_cvt_f32_i32_e32 v7, v7
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v227, v102, v88
	v_dual_mul_f32 v228, v98, v88 :: v_dual_mul_f32 v231, v100, v88
	v_mul_f32_e32 v110, v110, v72
	v_mul_f32_e32 v108, v108, v72
	v_mul_f32_e32 v104, v104, v72
	v_mul_f32_e32 v106, v106, v72
	v_mul_f32_e32 v102, v102, v72
	v_mul_f32_e32 v98, v98, v72
	v_mul_f32_e32 v100, v100, v72
	v_mul_f32_e32 v234, v96, v92
	v_mul_f32_e32 v236, v96, v88
	v_mul_f32_e32 v96, v96, v72
	v_mul_f32_e32 v238, v94, v90
	v_dual_mul_f32 v250, v94, v76 :: v_dual_mul_f32 v251, v94, v74
	v_dual_mul_f32 v66, v92, v76 :: v_dual_mul_f32 v193, v92, v74
	v_mul_f32_e32 v67, v88, v90
	v_mul_f32_e32 v196, v88, v86
	v_dual_mul_f32 v70, v88, v84 :: v_dual_mul_f32 v71, v88, v82
	v_mul_f32_e32 v194, v88, v80
	v_dual_mul_f32 v120, v88, v78 :: v_dual_mul_f32 v123, v88, v76
	v_mul_f32_e32 v124, v88, v74
	v_mul_f32_e32 v90, v72, v90
	v_mul_f32_e32 v86, v72, v86
	v_mul_f32_e32 v84, v72, v84
	v_mul_f32_e32 v82, v72, v82
	v_mul_f32_e32 v80, v72, v80
	v_mul_f32_e32 v78, v72, v78
	v_mul_f32_e32 v76, v72, v76
	v_mul_f32_e32 v74, v72, v74
	v_dual_mul_f32 v57, v111, v94 :: v_dual_mul_f32 v58, v109, v94
	v_dual_mul_f32 v59, v105, v94 :: v_dual_mul_f32 v60, v107, v94
	v_mul_f32_e32 v61, v103, v94
	v_dual_mul_f32 v62, v99, v94 :: v_dual_mul_f32 v63, v101, v94
	v_dual_mul_f32 v49, v111, v92 :: v_dual_mul_f32 v50, v109, v92
	v_dual_mul_f32 v51, v105, v92 :: v_dual_mul_f32 v52, v107, v92
	v_mul_f32_e32 v53, v103, v92
	v_dual_mul_f32 v54, v99, v92 :: v_dual_mul_f32 v55, v101, v92
	v_dual_mul_f32 v41, v111, v88 :: v_dual_mul_f32 v42, v109, v88
	v_dual_mul_f32 v225, v105, v88 :: v_dual_mul_f32 v226, v107, v88
	v_mul_f32_e32 v229, v103, v88
	v_mul_f32_e32 v230, v99, v88
	v_mul_f32_e32 v232, v101, v88
	v_mul_f32_e32 v111, v111, v72
	v_mul_f32_e32 v109, v109, v72
	v_mul_f32_e32 v105, v105, v72
	v_mul_f32_e32 v107, v107, v72
	v_mul_f32_e32 v103, v103, v72
	v_mul_f32_e32 v99, v99, v72
	v_mul_f32_e32 v101, v101, v72
	v_mul_f32_e32 v233, v97, v94
	v_mul_f32_e32 v235, v97, v92
	v_mul_f32_e32 v237, v97, v88
	v_dual_mul_f32 v97, v97, v72 :: v_dual_mul_f32 v240, v94, v91
	v_mul_f32_e32 v252, v94, v77
	v_dual_mul_f32 v94, v94, v75 :: v_dual_mul_f32 v195, v92, v77
	v_mul_f32_e32 v92, v92, v75
	v_cvt_f32_i32_e32 v89, v89
	v_mul_f32_e32 v68, v88, v91
	v_mul_f32_e32 v69, v88, v87
	v_dual_mul_f32 v187, v88, v85 :: v_dual_mul_f32 v188, v88, v83
	v_dual_mul_f32 v121, v88, v81 :: v_dual_mul_f32 v122, v88, v79
	v_dual_mul_f32 v125, v88, v77 :: v_dual_mul_f32 v88, v88, v75
	v_mul_f32_e32 v91, v72, v91
	v_mul_f32_e32 v87, v72, v87
	v_mul_f32_e32 v85, v72, v85
	v_mul_f32_e32 v83, v72, v83
	v_mul_f32_e32 v81, v72, v81
	v_mul_f32_e32 v79, v72, v79
	v_dual_mul_f32 v77, v72, v77 :: v_dual_mul_f32 v72, v72, v75
	v_cvt_f32_i32_e32 v73, v73
	v_dual_mul_f32 v75, v112, v114 :: v_dual_mul_f32 v112, v113, v115
	v_dual_mul_f32 v113, v116, v118 :: v_dual_mul_f32 v114, v117, v119
	v_dual_mul_f32 v115, v198, v200 :: v_dual_mul_f32 v116, v199, v201
	v_mul_f32_e32 v117, v202, v204
	v_mul_f32_e32 v64, v203, v64
	v_dual_mul_f32 v118, v205, v207 :: v_dual_mul_f32 v119, v206, v208
	v_dual_mul_f32 v198, v209, v211 :: v_dual_mul_f32 v199, v210, v212
	v_dual_mul_f32 v200, v213, v215 :: v_dual_mul_f32 v201, v214, v216
	v_mul_f32_e32 v202, v217, v218
	v_dual_mul_f32 v56, v234, v56 :: v_dual_mul_f32 v203, v219, v221
	v_dual_mul_f32 v204, v220, v222 :: v_dual_mul_f32 v43, v223, v43
	v_dual_mul_f32 v44, v224, v44 :: v_dual_mul_f32 v45, v227, v45
	v_dual_mul_f32 v46, v228, v46 :: v_dual_mul_f32 v47, v231, v47
	v_dual_mul_f32 v48, v236, v48 :: v_dual_mul_f32 v33, v110, v33
	v_mul_f32_e32 v34, v108, v34
	v_dual_mul_f32 v35, v104, v35 :: v_dual_mul_f32 v36, v106, v36
	v_mul_f32_e32 v37, v102, v37
	v_dual_mul_f32 v38, v98, v38 :: v_dual_mul_f32 v39, v100, v39
	v_dual_mul_f32 v40, v96, v40 :: v_dual_mul_f32 v25, v238, v25
	v_dual_mul_f32 v31, v250, v31 :: v_dual_mul_f32 v32, v251, v32
	v_dual_mul_f32 v23, v66, v23 :: v_dual_mul_f32 v24, v193, v24
	v_dual_mul_f32 v9, v67, v9 :: v_dual_mul_f32 v10, v196, v10
	v_dual_mul_f32 v11, v70, v11 :: v_dual_mul_f32 v12, v71, v12
	v_dual_mul_f32 v13, v194, v13 :: v_dual_mul_f32 v14, v120, v14
	v_dual_mul_f32 v15, v123, v15 :: v_dual_mul_f32 v16, v124, v16
	v_mul_f32_e32 v1, v90, v1
	v_dual_mul_f32 v2, v86, v2 :: v_dual_mul_f32 v3, v84, v3
	v_dual_mul_f32 v4, v82, v4 :: v_dual_mul_f32 v5, v80, v5
	v_dual_mul_f32 v6, v78, v6 :: v_dual_mul_f32 v7, v76, v7
	v_dual_mul_f32 v8, v74, v8 :: v_dual_fmac_f32 v75, v57, v95
	v_dual_fmac_f32 v112, v58, v95 :: v_dual_fmac_f32 v113, v59, v95
	v_dual_fmac_f32 v114, v60, v95 :: v_dual_fmac_f32 v115, v61, v95
	v_dual_fmac_f32 v116, v62, v95 :: v_dual_fmac_f32 v117, v63, v95
	v_fmac_f32_e32 v64, v233, v95
	v_dual_fmac_f32 v118, v49, v93 :: v_dual_fmac_f32 v119, v50, v93
	v_dual_fmac_f32 v198, v51, v93 :: v_dual_fmac_f32 v199, v52, v93
	v_dual_fmac_f32 v200, v53, v93 :: v_dual_fmac_f32 v201, v54, v93
	v_fmac_f32_e32 v202, v55, v93
	v_fmac_f32_e32 v56, v235, v93
	v_dual_fmac_f32 v203, v41, v89 :: v_dual_fmac_f32 v204, v42, v89
	v_dual_fmac_f32 v43, v225, v89 :: v_dual_fmac_f32 v44, v226, v89
	v_dual_fmac_f32 v45, v229, v89 :: v_dual_fmac_f32 v46, v230, v89
	v_dual_fmac_f32 v47, v232, v89 :: v_dual_fmac_f32 v48, v237, v89
	v_dual_fmac_f32 v33, v111, v73 :: v_dual_fmac_f32 v34, v109, v73
	v_dual_fmac_f32 v35, v105, v73 :: v_dual_fmac_f32 v36, v107, v73
	v_fmac_f32_e32 v37, v103, v73
	v_dual_fmac_f32 v38, v99, v73 :: v_dual_fmac_f32 v39, v101, v73
	v_dual_fmac_f32 v40, v97, v73 :: v_dual_fmac_f32 v25, v240, v95
	v_dual_fmac_f32 v31, v252, v95 :: v_dual_fmac_f32 v32, v94, v95
	v_dual_fmac_f32 v23, v195, v93 :: v_dual_fmac_f32 v24, v92, v93
	v_dual_fmac_f32 v9, v68, v89 :: v_dual_fmac_f32 v10, v69, v89
	v_dual_fmac_f32 v11, v187, v89 :: v_dual_fmac_f32 v12, v188, v89
	v_dual_fmac_f32 v13, v121, v89 :: v_dual_fmac_f32 v14, v122, v89
	v_dual_fmac_f32 v15, v125, v89 :: v_dual_fmac_f32 v16, v88, v89
	v_fmac_f32_e32 v1, v91, v73
	v_dual_fmac_f32 v2, v87, v73 :: v_dual_fmac_f32 v3, v85, v73
	v_dual_fmac_f32 v4, v83, v73 :: v_dual_fmac_f32 v5, v81, v73
	v_dual_fmac_f32 v6, v79, v73 :: v_dual_fmac_f32 v7, v77, v73
	v_dual_fmac_f32 v8, v72, v73 :: v_dual_add_f32 v179, v179, v75
	v_dual_add_f32 v186, v186, v112 :: v_dual_add_f32 v185, v185, v113
	v_add_f32_e32 v184, v184, v114
	v_dual_add_f32 v182, v182, v115 :: v_dual_add_f32 v183, v183, v116
	v_dual_add_f32 v180, v180, v117 :: v_dual_add_f32 v181, v181, v64
	v_dual_add_f32 v175, v175, v118 :: v_dual_add_f32 v176, v176, v119
	v_dual_add_f32 v173, v173, v198 :: v_dual_add_f32 v174, v174, v199
	v_dual_add_f32 v171, v171, v200 :: v_dual_add_f32 v172, v172, v201
	v_dual_add_f32 v169, v169, v202 :: v_dual_add_f32 v170, v170, v56
	v_dual_add_f32 v166, v166, v203 :: v_dual_add_f32 v167, v167, v204
	v_dual_add_f32 v164, v164, v43 :: v_dual_add_f32 v165, v165, v44
	v_dual_add_f32 v162, v162, v45 :: v_dual_add_f32 v163, v163, v46
	v_dual_add_f32 v160, v160, v47 :: v_dual_add_f32 v161, v161, v48
	v_dual_add_f32 v158, v158, v33 :: v_dual_add_f32 v159, v159, v34
	v_dual_add_f32 v156, v156, v35 :: v_dual_add_f32 v157, v157, v36
	v_dual_add_f32 v154, v154, v37 :: v_dual_add_f32 v155, v155, v38
	v_dual_add_f32 v152, v152, v39 :: v_dual_add_f32 v153, v153, v40
	v_add_f32_e32 v150, v150, v25
	v_dual_add_f32 v144, v144, v31 :: v_dual_add_f32 v145, v145, v32
	v_dual_add_f32 v136, v136, v23 :: v_dual_add_f32 v137, v137, v24
	v_dual_add_f32 v134, v134, v9 :: v_dual_add_f32 v135, v135, v10
	v_dual_add_f32 v132, v132, v11 :: v_dual_add_f32 v133, v133, v12
	v_add_f32_e32 v131, v131, v14
	s_wait_loadcnt 0xa
	v_add_f32_e32 v29, v29, v13
	s_wait_loadcnt 0x8
	v_dual_add_f32 v27, v27, v15 :: v_dual_add_f32 v28, v28, v16
	s_wait_loadcnt 0x7
	v_add_f32_e32 v30, v30, v1
	s_wait_loadcnt 0x5
	v_dual_add_f32 v26, v26, v2 :: v_dual_add_f32 v21, v21, v3
	s_wait_loadcnt 0x3
	v_dual_add_f32 v22, v22, v4 :: v_dual_add_f32 v19, v19, v5
	s_wait_loadcnt 0x2
	v_add_f32_e32 v20, v20, v6
	s_wait_loadcnt 0x0
	v_dual_add_f32 v18, v18, v7 :: v_dual_add_f32 v17, v17, v8
	v_mov_b32_e32 v68, v130
	s_xor_b32 s8, s26, -1
	s_mov_b32 s29, 1
	s_mov_b32 s26, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s8
	s_mov_b32 s28, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_7
.LBB3_10:                               ;   Parent Loop BB3_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s8, s29, s25
	s_lshl_b32 s30, s29, 6
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_mov_b32 s31, s9
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	v_add_co_u32 v1, s33, s30, v128
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v3, s36, s34, v126
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, s35, 0, s36
	v_add_co_ci_u32_e64 v2, null, s31, 0, s33
	v_add_co_u32 v5, s34, s34, v127
	v_add_co_u32 v7, s30, s30, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v6, null, s35, 0, s34
	global_load_b64 v[96:97], v[3:4], off offset:40
	v_add_co_ci_u32_e64 v8, null, s31, 0, s30
	global_load_b64 v[112:113], v[1:2], off offset:40
	v_add_nc_u32_e32 v1, s27, v68
	s_clause 0xa                            ; 44-byte Folded Spill
	scratch_store_b32 off, v30, off offset:40
	scratch_store_b32 off, v29, off offset:36
	scratch_store_b32 off, v28, off offset:32
	scratch_store_b32 off, v27, off offset:28
	scratch_store_b32 off, v26, off offset:24
	scratch_store_b32 off, v22, off offset:20
	scratch_store_b32 off, v21, off offset:16
	scratch_store_b32 off, v20, off offset:12
	scratch_store_b32 off, v19, off offset:8
	scratch_store_b32 off, v18, off offset:4
	scratch_store_b32 off, v17, off
	global_load_b64 v[98:99], v[5:6], off offset:40
	global_load_b64 v[114:115], v[7:8], off offset:40
	ds_load_2addr_b64 v[72:75], v197 offset1:32
	ds_load_2addr_b64 v[76:79], v1 offset1:32
	ds_load_2addr_b64 v[80:83], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[84:87], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[88:91], v197 offset0:192 offset1:224
	ds_load_2addr_b64 v[92:95], v1 offset0:64 offset1:96
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	scratch_load_b32 v0, off, off offset:44 ; 4-byte Folded Reload
	s_and_b32 s27, s28, s11
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s27
	v_wmma_i32_16x16x32_iu4 v[57:64], v[76:77], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[76:77], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[76:77], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[76:77], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[92:93], v[72:73], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[92:93], v[80:81], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[92:93], v[84:85], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[92:93], v[88:89], 0 neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[78:79], v[74:75], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[78:79], v[82:83], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[78:79], v[86:87], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[78:79], v[90:91], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[94:95], v[74:75], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[94:95], v[82:83], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[94:95], v[86:87], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[94:95], v[90:91], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_cndmask_b32_e64 v119, 0, v97, s0
	v_cndmask_b32_e64 v118, 0, v96, s0
	v_cndmask_b32_e64 v117, 0, v99, s1
	v_cndmask_b32_e64 v116, 0, v98, s1
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:16
	scratch_load_b32 v0, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:16
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_12
; %bb.11:                               ;   in Loop: Header=BB3_10 Depth=2
	s_clause 0x1                            ; 16-byte Folded Reload
	scratch_load_b32 v0, off, off offset:76
	scratch_load_b96 v[65:67], off, off offset:80
	s_add_co_i32 s8, s8, 1
	s_xor_b32 s33, s29, 1
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[8:9], s[14:15]
	s_add_co_i32 s8, s29, s10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[34:35], s[34:35], 0x48
	s_mul_u64 s[36:37], s[8:9], 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_mulk_i32 s8, 0x88
	s_lshl_b32 s29, s33, 2
	s_lshl_b32 s30, s33, 6
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v74, s33, s34, v127
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v75, null, s35, 0, s33
	s_mov_b32 s31, s9
	s_add_nc_u64 s[36:37], s[4:5], s[36:37]
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[30:31], s[36:37], s[30:31]
	s_wait_loadcnt 0x1
	v_mad_co_i64_i32 v[76:77], null, 0x48, v0, s[34:35]
	scratch_load_b32 v0, off, off offset:72 ; 4-byte Folded Reload
	s_wait_loadcnt 0x1
	v_add_co_u32 v78, vcc_lo, v65, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v66, vcc_lo
	v_add_co_u32 v72, s8, s34, v126
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v78, vcc_lo, v78, s29
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v73, null, s35, 0, s8
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v79, null, 0, v79, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v80, s8, s30, v128
	global_load_b64 v[118:119], v[72:73], off offset:8
	v_add_co_u32 v82, s29, s30, v129
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v81, null, s31, 0, s8
	v_add_co_ci_u32_e64 v83, null, s31, 0, s29
	s_wait_loadcnt 0x1
	v_add_co_u32 v72, vcc_lo, v76, v0
	global_load_b64 v[116:117], v[74:75], off offset:8
	global_load_b32 v0, v[78:79], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, 0, v77, vcc_lo
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:52 ; 4-byte Folded Spill
	s_clause 0x1
	global_load_b64 v[112:113], v[80:81], off offset:8
	global_load_b64 v[114:115], v[82:83], off offset:8
	global_load_b32 v0, v[72:73], off
	s_wait_loadcnt 0x0
	scratch_store_b32 off, v0, off offset:56 ; 4-byte Folded Spill
.LBB3_12:                               ; %.preheader468.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	scratch_load_b32 v0, off, off offset:60 ; 4-byte Folded Reload
	s_xor_b32 s8, s27, -1
	s_and_b32 s27, s26, exec_lo
	s_cselect_b32 s27, s23, 0x3c00
	v_add_nc_u32_e32 v72, 0, v68
	ds_load_2addr_b64 v[198:201], v197 offset1:32
	ds_load_2addr_b64 v[202:205], v197 offset0:64 offset1:96
	ds_load_2addr_b64 v[206:209], v197 offset0:128 offset1:160
	ds_load_2addr_b64 v[210:213], v197 offset0:192 offset1:224
	v_add_nc_u32_e32 v72, 0x2000, v72
	s_cselect_b32 s29, s13, 0x3400
	v_mov_b32_e32 v130, v68
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s8
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v74, s27, v0
	scratch_load_b32 v0, off, off offset:64 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x2000
	ds_load_2addr_b32 v[110:111], v74 offset1:1
	ds_load_2addr_b32 v[108:109], v74 offset0:2 offset1:3
	ds_load_2addr_b64 v[214:217], v72 offset1:32
	ds_load_2addr_b64 v[218:221], v72 offset0:64 offset1:96
	ds_load_2addr_b32 v[104:105], v74 offset0:4 offset1:5
	ds_load_2addr_b32 v[106:107], v74 offset0:6 offset1:7
	ds_load_2addr_b32 v[102:103], v74 offset0:8 offset1:9
	ds_load_2addr_b32 v[98:99], v74 offset0:10 offset1:11
	ds_load_2addr_b32 v[100:101], v74 offset0:12 offset1:13
	ds_load_2addr_b32 v[96:97], v74 offset0:14 offset1:15
	ds_load_2addr_b32 v[90:91], v74 offset0:32 offset1:33
	ds_load_2addr_b32 v[86:87], v74 offset0:34 offset1:35
	ds_load_2addr_b32 v[84:85], v74 offset0:36 offset1:37
	ds_load_2addr_b32 v[82:83], v74 offset0:38 offset1:39
	ds_load_2addr_b32 v[80:81], v74 offset0:40 offset1:41
	ds_load_2addr_b32 v[78:79], v74 offset0:42 offset1:43
	ds_load_2addr_b32 v[76:77], v74 offset0:44 offset1:45
	ds_load_2addr_b32 v[74:75], v74 offset0:46 offset1:47
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_wait_dscnt 0xf
	v_wmma_i32_16x16x32_iu4 v[57:64], v[214:215], v[198:199], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[214:215], v[202:203], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[214:215], v[206:207], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[214:215], v[210:211], v[33:40] neg_lo:[0,1,0]
	s_wait_dscnt 0xe
	v_wmma_i32_16x16x32_iu4 v[25:32], v[218:219], v[198:199], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[218:219], v[202:203], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[218:219], v[206:207], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[218:219], v[210:211], v[1:8] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[216:217], v[200:201], v[57:64] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[216:217], v[204:205], v[49:56] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[216:217], v[208:209], v[41:48] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[216:217], v[212:213], v[33:40] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[220:221], v[200:201], v[25:32] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[220:221], v[204:205], v[17:24] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[220:221], v[208:209], v[9:16] neg_lo:[0,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[220:221], v[212:213], v[1:8] neg_lo:[0,1,0]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v73, s29, v0
	ds_load_2addr_b32 v[94:95], v73 offset1:1
	ds_load_2addr_b32 v[92:93], v73 offset0:32 offset1:33
	ds_load_2addr_b32 v[88:89], v73 offset0:64 offset1:65
	ds_load_2addr_b32 v[72:73], v73 offset0:96 offset1:97
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB3_9
; %bb.13:                               ; %.preheader469.i
                                        ;   in Loop: Header=BB3_10 Depth=2
	s_clause 0x1                            ; 8-byte Folded Reload
	scratch_load_b32 v0, off, off offset:68
	scratch_load_b32 v65, off, off offset:52
	s_and_b32 s8, s28, exec_lo
	s_cselect_b32 s8, s13, 0x3400
	s_cselect_b32 s27, s23, 0x3c00
	v_cndmask_b32_e64 v119, 0, v119, s0
	v_cndmask_b32_e64 v118, 0, v118, s0
	v_cndmask_b32_e64 v117, 0, v117, s1
	v_cndmask_b32_e64 v116, 0, v116, s1
	s_wait_loadcnt 0x0
	v_lshrrev_b32_e32 v198, v0, v65
	scratch_load_b32 v0, off, off offset:56 ; 4-byte Folded Reload
	v_cvt_f32_f16_e64 v198, v198.l
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v198, 0, v198, s3
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v199, 0, v0, s2
	scratch_load_b32 v0, off, off offset:92 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v200, s8, v0
	v_add_nc_u32_e32 v201, s27, v0
	scratch_load_b32 v0, off, off offset:44 ; 4-byte Folded Reload
	s_movk_i32 s27, 0x1000
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[118:119], v[112:113] offset1:8
	scratch_load_b32 v0, off, off offset:48 ; 4-byte Folded Reload
	s_wait_loadcnt 0x0
	ds_store_2addr_stride64_b64 v0, v[116:117], v[114:115] offset1:8
	ds_store_b32 v200, v199
	ds_store_b32 v201, v198
	s_branch .LBB3_9
.LBB3_14:                               ; %Flow
	s_clause 0x2                            ; 12-byte Folded Reload
	scratch_load_b32 v0, off, off offset:96
	scratch_load_b32 v24, off, off offset:100
	scratch_load_b32 v25, off, off offset:104
	s_wait_loadcnt 0x2
	v_bfe_u32 v23, v0, 4, 1
.LBB3_15:                               ; %.preheader466.i
	v_lshrrev_b32_e32 v0, 4, v0
	s_wait_loadcnt 0x0
	v_mul_u32_u24_e32 v1, 0x500, v25
	v_lshlrev_b32_e32 v2, 2, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_b32_e32 v4, 15, v0
	v_or_b32_e32 v0, s21, v24
	v_add3_u32 v3, 0, v1, v2
	v_mul_u32_u24_e32 v2, 0x50, v24
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v5, 2, v4
	v_or_b32_e32 v4, s20, v4
	v_cmp_gt_i32_e64 s7, s12, v0
	v_mad_u32_u24 v6, 0x280, v23, v3
	v_ashrrev_i32_e32 v1, 31, v0
	v_add3_u32 v2, 0, v2, v5
	v_cmp_gt_i32_e32 vcc_lo, s14, v4
	ds_store_2addr_b32 v6, v179, v186 offset1:20
	ds_store_2addr_b32 v6, v185, v184 offset0:40 offset1:60
	ds_store_2addr_b32 v6, v182, v183 offset0:80 offset1:100
	ds_store_2addr_b32 v6, v180, v181 offset0:120 offset1:140
	s_wait_dscnt 0x0
	s_and_b32 s0, s7, vcc_lo
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB3_17
; %bb.16:
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
.LBB3_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_or_b32_e32 v5, 64, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s0, s14, v5
	s_and_b32 s1, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_19
; %bb.18:
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
.LBB3_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 32, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v6
	s_and_b32 s1, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_21
; %bb.20:
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
.LBB3_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_23
; %bb.22:
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
.LBB3_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 64, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s9, s12, v6
	s_and_b32 s1, s9, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_25
; %bb.24:
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
.LBB3_25:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_b32 s1, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_27
; %bb.26:
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
.LBB3_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_or_b32_e32 v6, 0x60, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v6
	s_and_b32 s1, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_29
; %bb.28:
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
.LBB3_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_mul_u32_u24_e32 v6, 0x280, v23
	s_and_b32 s1, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB3_31
; %bb.30:
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
.LBB3_31:                               ; %.preheader.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v3, v3, v6
	v_or_b32_e32 v6, 16, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s1, s14, v6
	ds_store_2addr_b32 v3, v175, v176 offset1:20
	ds_store_2addr_b32 v3, v173, v174 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v171, v172 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v169, v170 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s2, s7, s1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s2
	s_cbranch_execz .LBB3_33
; %bb.32:
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
.LBB3_33:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_or_b32_e32 v7, 0x50, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s2, s14, v7
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
.LBB3_41:                               ; %.preheader.2.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_or_b32_e32 v8, 32, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s3, s14, v8
	ds_store_2addr_b32 v3, v166, v167 offset1:20
	ds_store_2addr_b32 v3, v164, v165 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v162, v163 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v160, v161 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s4, s7, s3
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s5, s4
	s_cbranch_execz .LBB3_43
; %bb.42:
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
.LBB3_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s5
	v_or_b32_e32 v9, 0x60, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s4, s14, v9
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
.LBB3_51:                               ; %.preheader.3.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_or_b32_e32 v10, 48, v4
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s5, s14, v10
	ds_store_2addr_b32 v3, v158, v159 offset1:20
	ds_store_2addr_b32 v3, v156, v157 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v154, v155 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v152, v153 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s6, s7, s5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s6
	s_cbranch_execz .LBB3_53
; %bb.52:
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
.LBB3_53:
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v11, 0x70, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s6, s14, v11
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
.LBB3_61:                               ; %.preheader465.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_or_b32_e32 v12, 16, v0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	v_cmp_gt_i32_e64 s7, s12, v12
	ds_store_2addr_b32 v3, v150, v151 offset1:20
	ds_store_2addr_b32 v3, v148, v149 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v146, v147 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v144, v145 offset0:120 offset1:140
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_and_b32 s8, s7, vcc_lo
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_63
; %bb.62:
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
.LBB3_63:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_and_b32 s8, s7, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s9, s8
	s_cbranch_execz .LBB3_65
; %bb.64:
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
.LBB3_65:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	v_or_b32_e32 v12, 48, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s8, s12, v12
	s_and_b32 s9, s8, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_67
; %bb.66:
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
.LBB3_67:
	s_or_b32 exec_lo, exec_lo, s10
	s_and_b32 s9, s8, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s10, s9
	s_cbranch_execz .LBB3_69
; %bb.68:
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
.LBB3_69:
	s_or_b32 exec_lo, exec_lo, s10
	v_or_b32_e32 v12, 0x50, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_gt_i32_e64 s9, s12, v12
	s_and_b32 s10, s9, vcc_lo
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_71
; %bb.70:
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
.LBB3_71:
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s10, s9, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB3_73
; %bb.72:
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
.LBB3_73:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	v_or_b32_e32 v12, 0x70, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s10, s12, v12
	s_and_b32 s13, s10, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s11, s13
	s_cbranch_execz .LBB3_75
; %bb.74:
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
.LBB3_75:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_and_b32 s11, s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s0, s11
	s_cbranch_execz .LBB3_77
; %bb.76:
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
.LBB3_77:                               ; %.preheader.1.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s11, s7, s1
	s_wait_loadcnt 0x0
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
.LBB3_86:                               ; %.preheader.2.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s3
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v134, v135 offset1:20
	ds_store_2addr_b32 v3, v132, v133 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v29, v131 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v27, v28 offset0:120 offset1:140
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
.LBB3_95:                               ; %.preheader.3.1.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_and_b32 s1, s7, s5
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_2addr_b32 v3, v30, v26 offset1:20
	ds_store_2addr_b32 v3, v21, v22 offset0:40 offset1:60
	ds_store_2addr_b32 v3, v19, v20 offset0:80 offset1:100
	ds_store_2addr_b32 v3, v18, v17 offset0:120 offset1:140
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
	s_cbranch_execz .LBB3_35
.LBB3_107:
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
	s_cbranch_execz .LBB3_36
.LBB3_108:
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
	s_cbranch_execz .LBB3_37
.LBB3_109:
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
	s_cbranch_execz .LBB3_38
.LBB3_110:
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
	s_cbranch_execz .LBB3_39
.LBB3_111:
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
	s_cbranch_execnz .LBB3_40
	s_branch .LBB3_41
.LBB3_112:
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
	s_cbranch_execz .LBB3_45
.LBB3_113:
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
	s_cbranch_execz .LBB3_46
.LBB3_114:
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
	s_cbranch_execz .LBB3_47
.LBB3_115:
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
	s_cbranch_execz .LBB3_48
.LBB3_116:
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
	s_cbranch_execz .LBB3_49
.LBB3_117:
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
	s_cbranch_execnz .LBB3_50
	s_branch .LBB3_51
.LBB3_118:
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
	s_cbranch_execz .LBB3_55
.LBB3_119:
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
	s_cbranch_execz .LBB3_56
.LBB3_120:
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
	s_cbranch_execz .LBB3_57
.LBB3_121:
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
	s_cbranch_execz .LBB3_58
.LBB3_122:
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
	s_cbranch_execz .LBB3_59
.LBB3_123:
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
	s_cbranch_execnz .LBB3_60
	s_branch .LBB3_61
.LBB3_124:
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
	s_cbranch_execz .LBB3_79
.LBB3_125:
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
	s_cbranch_execz .LBB3_80
.LBB3_126:
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
	s_cbranch_execz .LBB3_81
.LBB3_127:
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
	s_cbranch_execz .LBB3_82
.LBB3_128:
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
	s_cbranch_execz .LBB3_83
.LBB3_129:
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
	s_cbranch_execz .LBB3_84
.LBB3_130:
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
	s_cbranch_execnz .LBB3_85
	s_branch .LBB3_86
.LBB3_131:
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
	s_cbranch_execz .LBB3_88
.LBB3_132:
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
	s_cbranch_execz .LBB3_89
.LBB3_133:
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
	s_cbranch_execz .LBB3_90
.LBB3_134:
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
	s_cbranch_execz .LBB3_91
.LBB3_135:
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
	s_cbranch_execz .LBB3_92
.LBB3_136:
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
	s_cbranch_execz .LBB3_93
.LBB3_137:
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
	s_cbranch_execnz .LBB3_94
	s_branch .LBB3_95
.LBB3_138:
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
	s_cbranch_execz .LBB3_97
.LBB3_139:
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
	s_cbranch_execz .LBB3_98
.LBB3_140:
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
	s_cbranch_execz .LBB3_99
.LBB3_141:
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
	s_cbranch_execz .LBB3_100
.LBB3_142:
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
	s_cbranch_execz .LBB3_101
.LBB3_143:
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
	s_cbranch_execz .LBB3_102
.LBB3_144:
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
	s_cbranch_execnz .LBB3_103
	s_branch .LBB3_104
.Lfunc_end3:
	.size	gemm_mq4g256v2_residual_mmq_iu4_full_set, .Lfunc_end3-gemm_mq4g256v2_residual_mmq_iu4_full_set
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_mmq_iu4_full_set
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 112
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
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.private_seg_size, 112
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.uses_flat_scratch, 1
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_mmq_iu4_full_set.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14284
; TotalNumSgprs: 40
; NumVgprs: 256
; ScratchSize: 112
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
	.type	__hip_cuid_c12ba5652c63586e,@object ; @__hip_cuid_c12ba5652c63586e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c12ba5652c63586e
__hip_cuid_c12ba5652c63586e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c12ba5652c63586e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c12ba5652c63586e
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
    .private_segment_fixed_size: 132
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 64
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
    .private_segment_fixed_size: 112
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 29
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
    .private_segment_fixed_size: 112
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_mmq_iu4_full_set.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 29
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
