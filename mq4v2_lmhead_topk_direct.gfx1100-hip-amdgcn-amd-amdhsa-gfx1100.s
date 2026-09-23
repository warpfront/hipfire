	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.amdhsa_code_object_version 6
	.text
	.protected	mq4v2_lmhead_topk_direct_gfx1100 ; -- Begin function mq4v2_lmhead_topk_direct_gfx1100
	.globl	mq4v2_lmhead_topk_direct_gfx1100
	.p2align	8
	.type	mq4v2_lmhead_topk_direct_gfx1100,@function
mq4v2_lmhead_topk_direct_gfx1100:       ; @mq4v2_lmhead_topk_direct_gfx1100
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x40
	s_load_b32 s33, s[0:1], 0x48
	s_mov_b32 s34, s15
	s_clause 0x1
	s_load_b256 s[8:15], s[0:1], 0x20
	s_load_b256 s[24:31], s[0:1], 0x0
	v_and_b32_e32 v39, 15, v0
	s_mov_b32 s36, 0xff800000
	s_mov_b32 s45, 0
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s16, s45
	s_mov_b32 s17, s45
	s_mov_b32 s18, s45
	s_mov_b32 s19, s45
	s_mov_b32 s20, s45
	s_waitcnt lgkmcnt(0)
	s_abs_i32 s3, s2
	s_mov_b32 s21, s45
	v_cvt_f32_u32_e32 v1, s3
	s_add_i32 s1, s12, 15
	v_cmp_gt_i32_e64 s0, s14, v39
	s_ashr_i32 s35, s1, 4
	s_mov_b32 s22, s45
	v_rcp_f32_e32 v1, v1
	s_mov_b32 s23, s45
	s_cmp_ge_i32 s34, s35
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s1, v1
	s_cbranch_scc1 .LBB0_298
; %bb.1:                                ; %.lr.ph554
	s_sub_i32 s4, 0, s3
	s_ashr_i32 s46, s13, 8
	s_mul_i32 s4, s4, s1
	s_abs_i32 s5, s46
	s_mul_hi_u32 s4, s1, s4
	s_ashr_i32 s2, s2, 31
	s_add_i32 s1, s1, s4
	s_ashr_i32 s4, s13, 31
	s_mul_hi_u32 s1, s5, s1
	s_xor_b32 s2, s4, s2
	s_mul_i32 s6, s1, s3
	v_cndmask_b32_e64 v4, 0, v39, s0
	s_sub_i32 s4, s5, s6
	s_add_i32 s5, s1, 1
	s_sub_i32 s6, s4, s3
	s_cmp_ge_u32 s4, s3
	v_lshrrev_b32_e32 v40, 4, v0
	s_cselect_b32 s1, s5, s1
	s_cselect_b32 s4, s6, s4
	s_add_i32 s5, s1, 1
	s_cmp_ge_u32 s4, s3
	v_mov_b32_e32 v33, 0xff800000
	s_cselect_b32 s1, s5, s1
	s_add_i32 s47, s12, -1
	s_xor_b32 s1, s1, s2
	v_mov_b32_e32 v34, 0
	s_sub_i32 s1, s1, s2
	s_cmp_gt_i32 s46, 0
	s_cselect_b32 s48, -1, 0
	s_cmp_gt_i32 s15, 1
	s_cselect_b32 s49, -1, 0
	s_cmp_gt_i32 s15, 2
	s_cselect_b32 s50, -1, 0
	s_cmp_gt_i32 s15, 3
	s_cselect_b32 s51, -1, 0
	s_cmp_gt_i32 s15, 4
	s_cselect_b32 s52, -1, 0
	s_cmp_gt_i32 s15, 5
	s_cselect_b32 s53, -1, 0
	s_cmp_gt_i32 s15, 6
	s_cselect_b32 s54, -1, 0
	s_cmp_gt_i32 s15, 7
	s_cselect_b32 s55, -1, 0
	s_abs_i32 s56, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cvt_f32_u32_e32 v1, s56
	s_sub_i32 s2, 0, s56
	v_rcp_f32_e32 v3, v1
	v_mad_i64_i32 v[1:2], null, v4, s13, 0
	s_mul_i32 s13, s46, 0x88
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v11, 0x4f7ffffe, v3
	v_dual_mov_b32 v3, s16 :: v_dual_mov_b32 v6, s19
	v_dual_mov_b32 v4, s17 :: v_dual_mov_b32 v5, s18
	v_mov_b32_e32 v8, s21
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_u32_f32_e32 v19, v11
	v_add_co_u32 v41, vcc_lo, s26, v1
	v_dual_mov_b32 v11, s36 :: v_dual_mov_b32 v14, s39
	v_readfirstlane_b32 s3, v19
	v_dual_mov_b32 v7, s20 :: v_dual_mov_b32 v10, s23
	v_dual_mov_b32 v9, s22 :: v_dual_mov_b32 v12, s37
	s_mul_i32 s2, s2, s3
	v_dual_mov_b32 v13, s38 :: v_dual_mov_b32 v16, s41
	v_dual_mov_b32 v15, s40 :: v_dual_mov_b32 v18, s43
	v_mov_b32_e32 v17, s42
	v_add_co_ci_u32_e64 v42, null, s27, v2, vcc_lo
	s_mul_hi_u32 s2, s3, s2
	s_ashr_i32 s26, s1, 31
	s_add_i32 s27, s3, s2
	s_mov_b32 s36, s34
	s_branch .LBB0_5
.LBB0_2:                                ; %Flow1604
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_3:                                ; %Flow1613
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s38
.LBB0_4:                                ; %Flow1684
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s37
	s_add_i32 s36, s36, s33
	s_cmp_ge_i32 s36, s35
	s_cbranch_scc1 .LBB0_299
.LBB0_5:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_8 Depth 2
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s48
	s_lshl_b32 s1, s36, 4
	s_cbranch_vccnz .LBB0_10
; %bb.6:                                ; %.lr.ph.preheader
                                        ;   in Loop: Header=BB0_5 Depth=1
	v_or_b32_e32 v1, s1, v39
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_i32_e32 v19, s47, v1
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_mad_i64_i32 v[1:2], null, s13, v19, s[24:25]
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, v50
	v_mov_b32_e32 v21, v50
	v_mov_b32_e32 v22, v50
	v_mov_b32_e32 v23, v50
	v_mov_b32_e32 v24, v50
	v_mov_b32_e32 v25, v50
	v_mov_b32_e32 v26, v50
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_8
.LBB0_7:                                ; %.loopexit519
                                        ;   in Loop: Header=BB0_8 Depth=2
	s_lshl_b32 s44, s2, 8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[4:5], s[44:45], 2
	v_add_co_u32 v35, vcc_lo, v41, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v36, null, s5, v42, vcc_lo
	s_mul_i32 s4, s2, 0x88
	s_add_i32 s2, s2, 1
	v_add_co_u32 v37, vcc_lo, v1, s4
	global_load_b128 v[27:30], v[35:36], off
	v_add_co_ci_u32_e64 v38, null, 0, v2, vcc_lo
	s_cmp_eq_u32 s2, s46
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v27
	v_cvt_f16_f32_e32 v51.h, v28
	v_cvt_f16_f32_e32 v52.l, v29
	v_cvt_f16_f32_e32 v52.h, v30
	global_load_b128 v[27:30], v[35:36], off offset:16
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v27
	v_cvt_f16_f32_e32 v53.h, v28
	v_cvt_f16_f32_e32 v54.l, v29
	v_cvt_f16_f32_e32 v54.h, v30
	global_load_b128 v[27:30], v[35:36], off offset:32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v27
	v_cvt_f16_f32_e32 v55.h, v28
	v_cvt_f16_f32_e32 v56.l, v29
	v_cvt_f16_f32_e32 v56.h, v30
	global_load_b128 v[27:30], v[35:36], off offset:48
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v27
	v_cvt_f16_f32_e32 v57.h, v28
	v_cvt_f16_f32_e32 v58.l, v29
	v_cvt_f16_f32_e32 v58.h, v30
	global_load_b128 v[27:30], v[37:38], off
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v31, 15, v29
	v_bfe_u32 v32, v29, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v31, v31
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v31.l, v31
	v_cvt_f16_f32_e32 v31.h, v32
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v59.l, v27.l, v31.l, v27.h
	v_fma_f16 v59.h, v27.l, v31.h, v27.h
	v_bfe_u32 v31, v29, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	v_cvt_f16_f32_e32 v31.l, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v60.l, v27.l, v31.l, v27.h
	v_bfe_u32 v31, v29, 12, 4
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v31.l, v31
	v_fma_f16 v60.h, v27.l, v31.l, v27.h
	v_bfe_u32 v31, v29, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	v_cvt_f16_f32_e32 v31.l, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v61.l, v27.l, v31.l, v27.h
	v_bfe_u32 v31, v29, 20, 4
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v31.l, v31
	v_fma_f16 v61.h, v27.l, v31.l, v27.h
	v_bfe_u32 v31, v29, 24, 4
	v_lshrrev_b32_e32 v29, 28, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v31, v31
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v31.l, v31
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v62.l, v27.l, v31.l, v27.h
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 4, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 12, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 20, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_lshrrev_b32_e32 v29, 28, v30
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	global_load_b128 v[29:32], v[35:36], off offset:64
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:80
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:96
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:112
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:16
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v59, 15, v29
	v_bfe_u32 v60, v29, 4, 4
	v_bfe_u32 v61, v29, 12, 4
	v_bfe_u32 v62, v29, 20, 4
	v_bfe_u32 v67, v32, 8, 4
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_bfe_u32 v68, v32, 12, 4
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	v_bfe_u32 v60, v29, 8, 4
	v_bfe_u32 v69, v32, 16, 4
	v_bfe_u32 v70, v32, 20, 4
	v_fma_f16 v59.l, v27.l, v59.l, v27.h
	v_fma_f16 v59.h, v27.l, v59.h, v27.h
	v_cvt_f32_ubyte0_e32 v60, v60
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	v_bfe_u32 v61, v29, 16, 4
	v_fma_f16 v60.l, v27.l, v60.l, v27.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v60.h, v27.l, v60.h, v27.h
	v_cvt_f32_ubyte0_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	v_bfe_u32 v62, v29, 24, 4
	v_lshrrev_b32_e32 v29, 28, v29
	v_fma_f16 v61.l, v27.l, v61.l, v27.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v61.h, v27.l, v61.h, v27.h
	v_cvt_f32_ubyte0_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v62.l, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v62.l, v27.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v30
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_lshrrev_b32_e32 v29, 28, v30
	v_bfe_u32 v30, v31, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:128
	v_cvt_f32_ubyte0_e32 v29, v29
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v29.l, v29
	v_cvt_f16_f32_e32 v29.h, v30
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:144
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:160
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:176
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.h, v60
	v_bfe_u32 v60, v31, 8, 4
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v27.l, v29.l, v27.h
	v_fma_f16 v59.h, v27.l, v29.h, v27.h
	v_lshrrev_b32_e32 v31, 28, v31
	v_cvt_f32_ubyte0_e32 v29, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	global_load_b128 v[29:32], v[35:36], off offset:192
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:208
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:224
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:240
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:32
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v59, 15, v29
	v_bfe_u32 v60, v29, 4, 4
	v_bfe_u32 v61, v29, 12, 4
	v_bfe_u32 v62, v29, 20, 4
	v_bfe_u32 v67, v32, 8, 4
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_bfe_u32 v68, v32, 12, 4
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	v_bfe_u32 v60, v29, 8, 4
	v_bfe_u32 v69, v32, 16, 4
	v_bfe_u32 v70, v32, 20, 4
	v_fma_f16 v59.l, v27.l, v59.l, v27.h
	v_fma_f16 v59.h, v27.l, v59.h, v27.h
	v_cvt_f32_ubyte0_e32 v60, v60
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	v_bfe_u32 v61, v29, 16, 4
	v_fma_f16 v60.l, v27.l, v60.l, v27.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v60.h, v27.l, v60.h, v27.h
	v_cvt_f32_ubyte0_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	v_bfe_u32 v62, v29, 24, 4
	v_lshrrev_b32_e32 v29, 28, v29
	v_fma_f16 v61.l, v27.l, v61.l, v27.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v61.h, v27.l, v61.h, v27.h
	v_cvt_f32_ubyte0_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v62.l, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v62.l, v27.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v30
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_lshrrev_b32_e32 v29, 28, v30
	v_bfe_u32 v30, v31, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:256
	v_cvt_f32_ubyte0_e32 v29, v29
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v29.l, v29
	v_cvt_f16_f32_e32 v29.h, v30
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:272
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:288
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:304
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.h, v60
	v_bfe_u32 v60, v31, 8, 4
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v27.l, v29.l, v27.h
	v_fma_f16 v59.h, v27.l, v29.h, v27.h
	v_lshrrev_b32_e32 v31, 28, v31
	v_cvt_f32_ubyte0_e32 v29, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	global_load_b128 v[29:32], v[35:36], off offset:320
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:336
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:352
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:368
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:48
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v59, 15, v29
	v_bfe_u32 v60, v29, 4, 4
	v_bfe_u32 v61, v29, 12, 4
	v_bfe_u32 v62, v29, 20, 4
	v_bfe_u32 v67, v32, 8, 4
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_bfe_u32 v68, v32, 12, 4
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	v_bfe_u32 v60, v29, 8, 4
	v_bfe_u32 v69, v32, 16, 4
	v_bfe_u32 v70, v32, 20, 4
	v_fma_f16 v59.l, v27.l, v59.l, v27.h
	v_fma_f16 v59.h, v27.l, v59.h, v27.h
	v_cvt_f32_ubyte0_e32 v60, v60
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	v_bfe_u32 v61, v29, 16, 4
	v_fma_f16 v60.l, v27.l, v60.l, v27.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v60.h, v27.l, v60.h, v27.h
	v_cvt_f32_ubyte0_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	v_bfe_u32 v62, v29, 24, 4
	v_lshrrev_b32_e32 v29, 28, v29
	v_fma_f16 v61.l, v27.l, v61.l, v27.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v61.h, v27.l, v61.h, v27.h
	v_cvt_f32_ubyte0_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v62.l, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v62.l, v27.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v30
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_lshrrev_b32_e32 v29, 28, v30
	v_bfe_u32 v30, v31, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v31
	s_delay_alu instid0(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:384
	v_cvt_f32_ubyte0_e32 v29, v29
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v29.l, v29
	v_cvt_f16_f32_e32 v29.h, v30
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:400
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:416
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:432
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.h, v60
	v_bfe_u32 v60, v31, 8, 4
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v27.l, v29.l, v27.h
	v_fma_f16 v59.h, v27.l, v29.h, v27.h
	v_lshrrev_b32_e32 v31, 28, v31
	v_cvt_f32_ubyte0_e32 v29, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v60.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v61.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_cvt_f32_ubyte0_e32 v29, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	global_load_b128 v[29:32], v[35:36], off offset:448
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:464
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:480
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:496
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:64
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v59, 15, v29
	v_bfe_u32 v60, v29, 4, 4
	v_bfe_u32 v61, v29, 12, 4
	v_bfe_u32 v62, v29, 20, 4
	v_bfe_u32 v67, v32, 8, 4
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v62, v62
	v_bfe_u32 v68, v32, 12, 4
	v_cvt_f16_f32_e32 v59.l, v59
	v_cvt_f16_f32_e32 v59.h, v60
	v_bfe_u32 v60, v29, 8, 4
	v_bfe_u32 v69, v32, 16, 4
	v_bfe_u32 v70, v32, 20, 4
	v_fma_f16 v59.l, v27.l, v59.l, v27.h
	v_fma_f16 v59.h, v27.l, v59.h, v27.h
	v_cvt_f32_ubyte0_e32 v60, v60
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v60.l, v60
	v_cvt_f16_f32_e32 v60.h, v61
	v_bfe_u32 v61, v29, 16, 4
	v_fma_f16 v60.l, v27.l, v60.l, v27.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v60.h, v27.l, v60.h, v27.h
	v_cvt_f32_ubyte0_e32 v61, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v61.l, v61
	v_cvt_f16_f32_e32 v61.h, v62
	v_bfe_u32 v62, v29, 24, 4
	v_lshrrev_b32_e32 v29, 28, v29
	v_fma_f16 v61.l, v27.l, v61.l, v27.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v61.h, v27.l, v61.h, v27.h
	v_cvt_f32_ubyte0_e32 v62, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v62.l, v62
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v62.l, v27.l, v62.l, v27.h
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v29, 15, v30
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v63.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v64.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v65.l, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v27.l, v29.l, v27.h
	v_bfe_u32 v29, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v29.l, v29
	v_fma_f16 v66.l, v27.l, v29.l, v27.h
	v_lshrrev_b32_e32 v29, 28, v30
	v_bfe_u32 v30, v31, 8, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v29.l, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_f16 v66.h, v27.l, v29.l, v27.h
	v_and_b32_e32 v27, 15, v31
	v_bfe_u32 v29, v31, 4, 4
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:512
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v29, v29
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v29
	v_bfe_u32 v64, v31, 24, 4
	v_lshrrev_b32_e32 v31, 28, v31
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:528
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:544
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:560
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_cvt_f32_ubyte0_e32 v27, v30
	v_cvt_f16_f32_e32 v58.h, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v32
	global_load_b128 v[29:32], v[35:36], off offset:576
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:592
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:608
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:624
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:80
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v27, 15, v29
	v_bfe_u32 v59, v29, 4, 4
	v_bfe_u32 v67, v32, 8, 4
	v_bfe_u32 v68, v32, 12, 4
	v_bfe_u32 v69, v32, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v59, v59
	v_bfe_u32 v70, v32, 20, 4
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_bfe_u32 v27, v29, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 12, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 20, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v29
	v_bfe_u32 v29, v31, 4, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v30
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v30
	v_bfe_u32 v30, v31, 8, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v31
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:640
	v_cvt_f32_ubyte0_e32 v27, v27
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v29
	v_lshrrev_b32_e32 v31, 28, v31
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:656
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:672
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:688
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_cvt_f32_ubyte0_e32 v27, v30
	v_cvt_f16_f32_e32 v58.h, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v32
	global_load_b128 v[29:32], v[35:36], off offset:704
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:720
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:736
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:752
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:96
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v27, 15, v29
	v_bfe_u32 v59, v29, 4, 4
	v_bfe_u32 v67, v32, 8, 4
	v_bfe_u32 v68, v32, 12, 4
	v_bfe_u32 v69, v32, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v59, v59
	v_bfe_u32 v70, v32, 20, 4
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_bfe_u32 v27, v29, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 12, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 20, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v29
	v_bfe_u32 v29, v31, 4, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v30
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v30
	v_bfe_u32 v30, v31, 8, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v31
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:768
	v_cvt_f32_ubyte0_e32 v27, v27
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v29
	v_lshrrev_b32_e32 v31, 28, v31
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:784
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:800
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:816
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_cvt_f32_ubyte0_e32 v27, v30
	v_cvt_f16_f32_e32 v58.h, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v32
	global_load_b128 v[29:32], v[35:36], off offset:832
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:848
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:864
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:880
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	global_load_b128 v[29:32], v[37:38], off offset:112
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v27, 15, v29
	v_bfe_u32 v59, v29, 4, 4
	v_bfe_u32 v67, v32, 8, 4
	v_bfe_u32 v68, v32, 12, 4
	v_bfe_u32 v69, v32, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v59, v59
	v_bfe_u32 v70, v32, 20, 4
	v_bfe_u32 v71, v32, 24, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_bfe_u32 v27, v29, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 12, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 20, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v29
	v_bfe_u32 v29, v31, 4, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v29, v29
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v30
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 8, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 16, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 24, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v30
	v_bfe_u32 v30, v31, 8, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v31
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	global_load_b128 v[51:54], v[35:36], off offset:896
	v_cvt_f32_ubyte0_e32 v27, v27
	v_bfe_u32 v61, v31, 12, 4
	v_bfe_u32 v62, v31, 16, 4
	v_bfe_u32 v63, v31, 20, 4
	v_bfe_u32 v64, v31, 24, 4
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v29
	v_lshrrev_b32_e32 v31, 28, v31
	v_and_b32_e32 v65, 15, v32
	v_bfe_u32 v66, v32, 4, 4
	v_lshrrev_b32_e32 v32, 28, v32
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v51
	v_cvt_f16_f32_e32 v51.h, v52
	v_cvt_f16_f32_e32 v52.l, v53
	v_cvt_f16_f32_e32 v52.h, v54
	global_load_b128 v[53:56], v[35:36], off offset:912
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v53
	v_cvt_f16_f32_e32 v53.h, v54
	v_cvt_f16_f32_e32 v54.l, v55
	v_cvt_f16_f32_e32 v54.h, v56
	global_load_b128 v[55:58], v[35:36], off offset:928
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v55
	v_cvt_f16_f32_e32 v55.h, v56
	v_cvt_f16_f32_e32 v56.l, v57
	v_cvt_f16_f32_e32 v56.h, v58
	global_load_b128 v[57:60], v[35:36], off offset:944
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v57
	v_cvt_f16_f32_e32 v57.h, v58
	v_cvt_f16_f32_e32 v58.l, v59
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_cvt_f32_ubyte0_e32 v27, v30
	v_cvt_f16_f32_e32 v58.h, v60
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v68
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v69
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v32
	global_load_b128 v[29:32], v[35:36], off offset:960
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f16 v66.h, v28.l, v27.l, v28.h
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v51.l, v29
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v52.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:976
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v53.l, v29
	v_cvt_f16_f32_e32 v53.h, v30
	v_cvt_f16_f32_e32 v54.l, v31
	v_cvt_f16_f32_e32 v54.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:992
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v55.l, v29
	v_cvt_f16_f32_e32 v55.h, v30
	v_cvt_f16_f32_e32 v56.l, v31
	v_cvt_f16_f32_e32 v56.h, v32
	global_load_b128 v[29:32], v[35:36], off offset:1008
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v57.l, v29
	v_cvt_f16_f32_e32 v57.h, v30
	global_load_b64 v[29:30], v[37:38], off offset:128
	v_cvt_f16_f32_e32 v58.l, v31
	v_cvt_f16_f32_e32 v58.h, v32
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v27, 15, v29
	v_bfe_u32 v31, v29, 4, 4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v27.l, v27
	v_cvt_f16_f32_e32 v27.h, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v59.l, v28.l, v27.l, v28.h
	v_fma_f16 v59.h, v28.l, v27.h, v28.h
	v_bfe_u32 v27, v29, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v60.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 12, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v60.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v61.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 20, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v61.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v29, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v62.l, v28.l, v27.l, v28.h
	v_lshrrev_b32_e32 v27, 28, v29
	v_lshrrev_b32_e32 v29, 28, v30
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v62.h, v28.l, v27.l, v28.h
	v_and_b32_e32 v27, 15, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v63.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 4, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v63.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v64.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 12, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v64.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v65.l, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 20, 4
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27.l, v27
	v_fma_f16 v65.h, v28.l, v27.l, v28.h
	v_bfe_u32 v27, v30, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v66.l, v28.l, v27.l, v28.h
	v_cvt_f32_ubyte0_e32 v27, v29
	v_cvt_f16_f32_e32 v27.l, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f16_e32 v28.h, v28.l, v27.l
	v_mov_b16_e32 v66.h, v28.h
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[19:26], v[59:66], v[51:58], v[19:26]
	s_cbranch_scc1 .LBB0_10
.LBB0_8:                                ; %.lr.ph
                                        ;   Parent Loop BB0_5 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_mul_hi_u32 s4, s2, s27
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s4, s56
	s_add_i32 s6, s4, 1
	s_sub_i32 s5, s2, s5
	s_sub_i32 s7, s5, s56
	s_cmp_ge_u32 s5, s56
	s_cselect_b32 s4, s6, s4
	s_cselect_b32 s5, s7, s5
	s_add_i32 s6, s4, 1
	s_cmp_ge_u32 s5, s56
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s4, s4, s26
	s_sub_i32 s4, s4, s26
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_le_i32 s4, s3
	s_cbranch_scc1 .LBB0_7
; %bb.9:                                ; %.preheader518.preheader
                                        ;   in Loop: Header=BB0_8 Depth=2
	v_dual_add_f32 v50, v19, v50 :: v_dual_add_f32 v49, v20, v49
	v_dual_mov_b32 v19, 0 :: v_dual_add_f32 v46, v23, v46
	v_dual_add_f32 v48, v21, v48 :: v_dual_add_f32 v47, v22, v47
	v_dual_add_f32 v45, v24, v45 :: v_dual_add_f32 v44, v25, v44
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v43, v26, v43 :: v_dual_mov_b32 v20, v19
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_mov_b32_e32 v24, v19
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v26, v19
	s_mov_b32 s3, s4
	s_branch .LBB0_7
.LBB0_10:                               ; %Flow1686
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_saveexec_b32 s37, s0
	s_cbranch_execz .LBB0_4
; %bb.11:                               ; %.preheader520
                                        ;   in Loop: Header=BB0_5 Depth=1
	v_add_nc_u32_e32 v1, s1, v40
	s_mov_b32 s38, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s12, v1
	s_cbranch_execz .LBB0_47
; %bb.12:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v2, v19, v50
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v2, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_18
; %bb.13:                               ; %Flow1682
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_19
.LBB0_14:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_20
.LBB0_15:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v27, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.16:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 1
	v_mov_b32_e32 v27, v12
; %bb.17:                               ; %Flow1681
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_21
	s_branch .LBB0_24
.LBB0_18:                               ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v19, v2, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v27, 0x3fb8aa3b, v19
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	v_fma_f32 v28, 0x3fb8aa3b, v19, -v27
	v_rndne_f32_e32 v29, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, 0x32a5705f, v19 :: v_dual_sub_f32 v27, v27, v29
	v_add_f32_e32 v27, v27, v28
	v_cvt_i32_f32_e32 v28, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v27, v27
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v27, v27, v28
	v_cndmask_b32_e32 v27, 0, v27, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v19, 0x7f800000, v27, vcc_lo
	v_add_f32_e32 v34, v34, v19
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_14
.LBB0_19:                               ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v19, v33, v2
	v_mov_b32_e32 v33, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v27, 0x3fb8aa3b, v19
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	v_fma_f32 v28, 0x3fb8aa3b, v19, -v27
	v_rndne_f32_e32 v29, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, 0x32a5705f, v19 :: v_dual_sub_f32 v27, v27, v29
	v_add_f32_e32 v27, v27, v28
	v_cvt_i32_f32_e32 v28, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v27, v27
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v27, v27, v28
	v_cndmask_b32_e32 v27, 0, v27, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v19, 0x7f800000, v27, vcc_lo
	v_fma_f32 v34, v34, v19, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_15
.LBB0_20:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v27, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_24
.LBB0_21:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v27
; %bb.22:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 2
	v_mov_b32_e32 v27, v13
; %bb.23:                               ; %Flow1680
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_24:                               ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_29
; %bb.25:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_32
.LBB0_26:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_35
.LBB0_27:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_38
.LBB0_28:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_41
	s_branch .LBB0_44
.LBB0_29:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v27
; %bb.30:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 3
	v_mov_b32_e32 v27, v14
; %bb.31:                               ; %Flow1679
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_26
.LBB0_32:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v27
; %bb.33:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 4
	v_mov_b32_e32 v27, v15
; %bb.34:                               ; %Flow1678
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_27
.LBB0_35:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v27
; %bb.36:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 5
	v_mov_b32_e32 v27, v16
; %bb.37:                               ; %Flow1677
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_28
.LBB0_38:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v27
; %bb.39:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 6
	v_mov_b32_e32 v27, v17
; %bb.40:                               ; %Flow1676
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_44
.LBB0_41:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v27
; %bb.42:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v19, 7
	v_mov_b32_e32 v27, v18
; %bb.43:                               ; %Flow1675
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_44:                               ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v2, v27
	s_cbranch_execz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v19
	v_cmp_eq_u32_e64 s1, 6, v19
	v_cmp_eq_u32_e64 s2, 5, v19
	v_cmp_eq_u32_e64 s3, 4, v19
	v_cmp_eq_u32_e64 s4, 3, v19
	v_cmp_eq_u32_e64 s5, 2, v19
	v_cmp_eq_u32_e64 s6, 1, v19
	v_cmp_eq_u32_e64 s7, 0, v19
	v_cndmask_b32_e32 v18, v18, v2, vcc_lo
	v_cndmask_b32_e64 v17, v17, v2, s1
	v_cndmask_b32_e64 v16, v16, v2, s2
	v_cndmask_b32_e64 v15, v15, v2, s3
	v_cndmask_b32_e64 v14, v14, v2, s4
	v_cndmask_b32_e64 v13, v13, v2, s5
	v_cndmask_b32_e64 v12, v12, v2, s6
	v_cndmask_b32_e64 v11, v11, v2, s7
	v_cndmask_b32_e32 v10, v10, v1, vcc_lo
	v_cndmask_b32_e64 v9, v9, v1, s1
	v_cndmask_b32_e64 v8, v8, v1, s2
	v_cndmask_b32_e64 v7, v7, v1, s3
	v_cndmask_b32_e64 v6, v6, v1, s4
	v_cndmask_b32_e64 v5, v5, v1, s5
	v_cndmask_b32_e64 v4, v4, v1, s6
	v_cndmask_b32_e64 v3, v3, v1, s7
.LBB0_46:                               ; %Flow1674
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_47:                               ; %Flow1683
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 2, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_83
; %bb.48:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v20, v49
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_54
; %bb.49:                               ; %Flow1672
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_55
.LBB0_50:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_56
.LBB0_51:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v27, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.52:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v27, v12
; %bb.53:                               ; %Flow1671
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_57
	s_branch .LBB0_60
.LBB0_54:                               ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v27, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v28, 0x3fb8aa3b, v20, -v27
	v_rndne_f32_e32 v29, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v28, 0x32a5705f, v20 :: v_dual_sub_f32 v27, v27, v29
	v_add_f32_e32 v27, v27, v28
	v_cvt_i32_f32_e32 v28, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v27, v27
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v27, v27, v28
	v_cndmask_b32_e32 v27, 0, v27, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v20, 0x7f800000, v27, vcc_lo
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_50
.LBB0_55:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v27, 0x3fb8aa3b, v20
	v_fma_f32 v28, 0x3fb8aa3b, v20, -v27
	v_rndne_f32_e32 v29, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v27, v27, v29
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fmac_f32_e32 v28, 0x32a5705f, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v27, v27, v28
	v_cvt_i32_f32_e32 v28, v29
	v_exp_f32_e32 v27, v27
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v27, v27, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v27, 0, v27, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v27, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_51
.LBB0_56:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v27, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_60
.LBB0_57:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v27
; %bb.58:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v27, v13
; %bb.59:                               ; %Flow1670
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_60:                               ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_65
; %bb.61:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_68
.LBB0_62:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_71
.LBB0_63:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_74
.LBB0_64:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_77
	s_branch .LBB0_80
.LBB0_65:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v27
; %bb.66:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v27, v14
; %bb.67:                               ; %Flow1669
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_62
.LBB0_68:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v27
; %bb.69:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v27, v15
; %bb.70:                               ; %Flow1668
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_63
.LBB0_71:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v27
; %bb.72:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v27, v16
; %bb.73:                               ; %Flow1667
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_64
.LBB0_74:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v27
; %bb.75:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v27, v17
; %bb.76:                               ; %Flow1666
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_80
.LBB0_77:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v27
; %bb.78:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v27, v18
; %bb.79:                               ; %Flow1665
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_80:                               ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v27
	s_cbranch_execz .LBB0_82
; %bb.81:                               ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_82:                               ; %Flow1664
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_83:                               ; %Flow1673
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 4, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_119
; %bb.84:                               ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v21, v48
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_90
; %bb.85:                               ; %Flow1662
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_91
.LBB0_86:                               ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_92
.LBB0_87:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.88:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v21, v12
; %bb.89:                               ; %Flow1661
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_93
	s_branch .LBB0_96
.LBB0_90:                               ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_fma_f32 v27, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v28, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v21, v21, v28
	v_fmac_f32_e32 v27, 0x32a5705f, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v27
	v_cvt_i32_f32_e32 v27, v28
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_86
.LBB0_91:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v27, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v28, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v27, 0x32a5705f, v20
	v_sub_f32_e32 v21, v21, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v27
	v_cvt_i32_f32_e32 v27, v28
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_87
.LBB0_92:                               ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v21, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_96
.LBB0_93:                               ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v21
; %bb.94:                               ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v21, v13
; %bb.95:                               ; %Flow1660
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_96:                               ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_101
; %bb.97:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_104
.LBB0_98:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_107
.LBB0_99:                               ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_110
.LBB0_100:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_113
	s_branch .LBB0_116
.LBB0_101:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v21
; %bb.102:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v21, v14
; %bb.103:                              ; %Flow1659
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_98
.LBB0_104:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v21
; %bb.105:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v21, v15
; %bb.106:                              ; %Flow1658
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_99
.LBB0_107:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v21
; %bb.108:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v21, v16
; %bb.109:                              ; %Flow1657
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_100
.LBB0_110:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v21
; %bb.111:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v21, v17
; %bb.112:                              ; %Flow1656
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_116
.LBB0_113:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v21
; %bb.114:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v21, v18
; %bb.115:                              ; %Flow1655
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_116:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v21
	s_cbranch_execz .LBB0_118
; %bb.117:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_118:                              ; %Flow1654
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_119:                              ; %Flow1663
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 6, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_155
; %bb.120:                              ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v22, v47
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_126
; %bb.121:                              ; %Flow1652
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_127
.LBB0_122:                              ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_128
.LBB0_123:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.124:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v21, v12
; %bb.125:                              ; %Flow1651
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_129
	s_branch .LBB0_132
.LBB0_126:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v27, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v21, v21, v27
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fmac_f32_e32 v22, 0x32a5705f, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v27
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_122
.LBB0_127:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v27, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, 0x32a5705f, v20 :: v_dual_sub_f32 v21, v21, v27
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v27
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_123
.LBB0_128:                              ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v21, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_132
.LBB0_129:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v21
; %bb.130:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v21, v13
; %bb.131:                              ; %Flow1650
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_132:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_137
; %bb.133:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_140
.LBB0_134:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_143
.LBB0_135:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_146
.LBB0_136:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_149
	s_branch .LBB0_152
.LBB0_137:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v21
; %bb.138:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v21, v14
; %bb.139:                              ; %Flow1649
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_134
.LBB0_140:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v21
; %bb.141:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v21, v15
; %bb.142:                              ; %Flow1648
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_135
.LBB0_143:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v21
; %bb.144:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v21, v16
; %bb.145:                              ; %Flow1647
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_136
.LBB0_146:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v21
; %bb.147:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v21, v17
; %bb.148:                              ; %Flow1646
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_152
.LBB0_149:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v21
; %bb.150:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v21, v18
; %bb.151:                              ; %Flow1645
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_152:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v21
	s_cbranch_execz .LBB0_154
; %bb.153:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_154:                              ; %Flow1644
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_155:                              ; %Flow1653
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 8, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_191
; %bb.156:                              ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v23, v46
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_162
; %bb.157:                              ; %Flow1642
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_163
.LBB0_158:                              ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_164
.LBB0_159:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.160:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v21, v12
; %bb.161:                              ; %Flow1641
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_165
	s_branch .LBB0_168
.LBB0_162:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v21, v21, v23
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fmac_f32_e32 v22, 0x32a5705f, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_158
.LBB0_163:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, 0x32a5705f, v20 :: v_dual_sub_f32 v21, v21, v23
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_159
.LBB0_164:                              ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v21, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_168
.LBB0_165:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v21
; %bb.166:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v21, v13
; %bb.167:                              ; %Flow1640
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_168:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_173
; %bb.169:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_176
.LBB0_170:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_179
.LBB0_171:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_182
.LBB0_172:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_185
	s_branch .LBB0_188
.LBB0_173:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v21
; %bb.174:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v21, v14
; %bb.175:                              ; %Flow1639
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_170
.LBB0_176:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v21
; %bb.177:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v21, v15
; %bb.178:                              ; %Flow1638
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_171
.LBB0_179:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v21
; %bb.180:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v21, v16
; %bb.181:                              ; %Flow1637
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_172
.LBB0_182:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v21
; %bb.183:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v21, v17
; %bb.184:                              ; %Flow1636
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_188
.LBB0_185:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v21
; %bb.186:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v21, v18
; %bb.187:                              ; %Flow1635
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_188:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v21
	s_cbranch_execz .LBB0_190
; %bb.189:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_190:                              ; %Flow1634
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_191:                              ; %Flow1643
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 10, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_227
; %bb.192:                              ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v24, v45
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_198
; %bb.193:                              ; %Flow1632
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_199
.LBB0_194:                              ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_200
.LBB0_195:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.196:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v21, v12
; %bb.197:                              ; %Flow1631
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_201
	s_branch .LBB0_204
.LBB0_198:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v21, v21, v23
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fmac_f32_e32 v22, 0x32a5705f, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_194
.LBB0_199:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, 0x32a5705f, v20 :: v_dual_sub_f32 v21, v21, v23
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_195
.LBB0_200:                              ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v21, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_204
.LBB0_201:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v21
; %bb.202:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v21, v13
; %bb.203:                              ; %Flow1630
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_204:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_209
; %bb.205:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_212
.LBB0_206:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_215
.LBB0_207:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_218
.LBB0_208:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_221
	s_branch .LBB0_224
.LBB0_209:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v21
; %bb.210:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v21, v14
; %bb.211:                              ; %Flow1629
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_206
.LBB0_212:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v21
; %bb.213:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v21, v15
; %bb.214:                              ; %Flow1628
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_207
.LBB0_215:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v21
; %bb.216:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v21, v16
; %bb.217:                              ; %Flow1627
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_208
.LBB0_218:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v21
; %bb.219:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v21, v17
; %bb.220:                              ; %Flow1626
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_224
.LBB0_221:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v21
; %bb.222:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v21, v18
; %bb.223:                              ; %Flow1625
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_224:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v21
	s_cbranch_execz .LBB0_226
; %bb.225:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_226:                              ; %Flow1624
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_227:                              ; %Flow1633
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v2, 12, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v2
	s_cbranch_execz .LBB0_263
; %bb.228:                              ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v19, v25, v44
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v19, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_234
; %bb.229:                              ; %Flow1622
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_235
.LBB0_230:                              ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_236
.LBB0_231:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.232:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 1 :: v_dual_mov_b32 v21, v12
; %bb.233:                              ; %Flow1621
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_237
	s_branch .LBB0_240
.LBB0_234:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v20, v19, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v21, v21, v23
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fmac_f32_e32 v22, 0x32a5705f, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v20
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_230
.LBB0_235:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_sub_f32 v20, v33, v19 :: v_dual_mov_b32 v33, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v22, 0x3fb8aa3b, v20, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, 0x32a5705f, v20 :: v_dual_sub_f32 v21, v21, v23
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v20, 0x7f800000, v21, vcc_lo
	v_fma_f32 v34, v34, v20, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v20, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_231
.LBB0_236:                              ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v21, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_240
.LBB0_237:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v21
; %bb.238:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 2 :: v_dual_mov_b32 v21, v13
; %bb.239:                              ; %Flow1620
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_240:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_245
; %bb.241:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_248
.LBB0_242:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_251
.LBB0_243:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_254
.LBB0_244:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_257
	s_branch .LBB0_260
.LBB0_245:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v21
; %bb.246:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 3 :: v_dual_mov_b32 v21, v14
; %bb.247:                              ; %Flow1619
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_242
.LBB0_248:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v21
; %bb.249:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 4 :: v_dual_mov_b32 v21, v15
; %bb.250:                              ; %Flow1618
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_243
.LBB0_251:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v21
; %bb.252:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 5 :: v_dual_mov_b32 v21, v16
; %bb.253:                              ; %Flow1617
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_244
.LBB0_254:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v21
; %bb.255:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 6 :: v_dual_mov_b32 v21, v17
; %bb.256:                              ; %Flow1616
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_260
.LBB0_257:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v21
; %bb.258:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v20, 7 :: v_dual_mov_b32 v21, v18
; %bb.259:                              ; %Flow1615
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_260:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v19, v21
	s_cbranch_execz .LBB0_262
; %bb.261:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v20
	v_cmp_eq_u32_e64 s1, 6, v20
	v_cmp_eq_u32_e64 s2, 5, v20
	v_cmp_eq_u32_e64 s3, 4, v20
	v_cmp_eq_u32_e64 s4, 3, v20
	v_cmp_eq_u32_e64 s5, 2, v20
	v_cmp_eq_u32_e64 s6, 1, v20
	v_cmp_eq_u32_e64 s7, 0, v20
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s1
	v_cndmask_b32_e64 v16, v16, v19, s2
	v_cndmask_b32_e64 v15, v15, v19, s3
	v_cndmask_b32_e64 v14, v14, v19, s4
	v_cndmask_b32_e64 v13, v13, v19, s5
	v_cndmask_b32_e64 v12, v12, v19, s6
	v_cndmask_b32_e64 v11, v11, v19, s7
	v_cndmask_b32_e32 v10, v10, v2, vcc_lo
	v_cndmask_b32_e64 v9, v9, v2, s1
	v_cndmask_b32_e64 v8, v8, v2, s2
	v_cndmask_b32_e64 v7, v7, v2, s3
	v_cndmask_b32_e64 v6, v6, v2, s4
	v_cndmask_b32_e64 v5, v5, v2, s5
	v_cndmask_b32_e64 v4, v4, v2, s6
	v_cndmask_b32_e64 v3, v3, v2, s7
.LBB0_262:                              ; %Flow1614
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_263:                              ; %Flow1623
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_or_b32 exec_lo, exec_lo, s38
	v_add_nc_u32_e32 v1, 14, v1
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_i32_e64 s12, v1
	s_cbranch_execz .LBB0_3
; %bb.264:                              ;   in Loop: Header=BB0_5 Depth=1
	v_add_f32_e32 v2, v26, v43
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_ngt_f32_e32 v2, v33
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execnz .LBB0_270
; %bb.265:                              ; %Flow1612
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execnz .LBB0_271
.LBB0_266:                              ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccnz .LBB0_272
.LBB0_267:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, v11
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v12, v11
; %bb.268:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 1 :: v_dual_mov_b32 v20, v12
; %bb.269:                              ; %Flow1611
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccz .LBB0_273
	s_branch .LBB0_276
.LBB0_270:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v19, v2, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v20, 0x3fb8aa3b, v19
	v_fma_f32 v21, 0x3fb8aa3b, v19, -v20
	v_rndne_f32_e32 v22, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v20, v20, v22
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	v_fmac_f32_e32 v21, 0x32a5705f, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v20, v21
	v_cvt_i32_f32_e32 v21, v22
	v_exp_f32_e32 v20, v20
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v20, v20, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v20, 0, v20, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	v_cndmask_b32_e32 v19, 0x7f800000, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v34, v34, v19
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_266
.LBB0_271:                              ;   in Loop: Header=BB0_5 Depth=1
	v_sub_f32_e32 v19, v33, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v33, v2 :: v_dual_mul_f32 v20, 0x3fb8aa3b, v19
	v_fma_f32 v21, 0x3fb8aa3b, v19, -v20
	v_rndne_f32_e32 v22, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v21, 0x32a5705f, v19 :: v_dual_sub_f32 v20, v20, v22
	v_add_f32_e32 v20, v20, v21
	v_cvt_i32_f32_e32 v21, v22
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v20, v20
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v20, v20, v21
	v_cndmask_b32_e32 v20, 0, v20, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v19, 0x7f800000, v20, vcc_lo
	v_fma_f32 v34, v34, v19, 1.0
	s_or_b32 exec_lo, exec_lo, s1
	v_mov_b32_e32 v19, 0
	s_and_not1_b32 vcc_lo, exec_lo, s49
	s_cbranch_vccz .LBB0_267
.LBB0_272:                              ;   in Loop: Header=BB0_5 Depth=1
	v_mov_b32_e32 v20, v11
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_cbranch_vccnz .LBB0_276
.LBB0_273:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_lt_f32_e32 v13, v20
; %bb.274:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 2 :: v_dual_mov_b32 v20, v13
; %bb.275:                              ; %Flow1610
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_276:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s51
	s_cbranch_vccz .LBB0_281
; %bb.277:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccz .LBB0_284
.LBB0_278:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccz .LBB0_287
.LBB0_279:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccz .LBB0_290
.LBB0_280:                              ;   in Loop: Header=BB0_5 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccz .LBB0_293
	s_branch .LBB0_296
.LBB0_281:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v14, v20
; %bb.282:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 3 :: v_dual_mov_b32 v20, v14
; %bb.283:                              ; %Flow1609
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s52
	s_cbranch_vccnz .LBB0_278
.LBB0_284:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v15, v20
; %bb.285:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 4 :: v_dual_mov_b32 v20, v15
; %bb.286:                              ; %Flow1608
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s53
	s_cbranch_vccnz .LBB0_279
.LBB0_287:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v16, v20
; %bb.288:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 5 :: v_dual_mov_b32 v20, v16
; %bb.289:                              ; %Flow1607
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_cbranch_vccnz .LBB0_280
.LBB0_290:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v17, v20
; %bb.291:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 6 :: v_dual_mov_b32 v20, v17
; %bb.292:                              ; %Flow1606
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s55
	s_cbranch_vccnz .LBB0_296
.LBB0_293:                              ;   in Loop: Header=BB0_5 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_lt_f32_e32 v18, v20
; %bb.294:                              ;   in Loop: Header=BB0_5 Depth=1
	v_dual_mov_b32 v19, 7 :: v_dual_mov_b32 v20, v18
; %bb.295:                              ; %Flow1605
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
.LBB0_296:                              ;   in Loop: Header=BB0_5 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_mov_b32 s39, exec_lo
	v_cmpx_gt_f32_e32 v2, v20
	s_cbranch_execz .LBB0_2
; %bb.297:                              ;   in Loop: Header=BB0_5 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 7, v19
	v_cmp_eq_u32_e64 s1, 6, v19
	v_cmp_eq_u32_e64 s2, 5, v19
	v_cmp_eq_u32_e64 s3, 4, v19
	v_cmp_eq_u32_e64 s4, 3, v19
	v_cmp_eq_u32_e64 s5, 2, v19
	v_cmp_eq_u32_e64 s6, 1, v19
	v_cmp_eq_u32_e64 s7, 0, v19
	v_cndmask_b32_e32 v18, v18, v2, vcc_lo
	v_cndmask_b32_e64 v17, v17, v2, s1
	v_cndmask_b32_e64 v16, v16, v2, s2
	v_cndmask_b32_e64 v15, v15, v2, s3
	v_cndmask_b32_e64 v14, v14, v2, s4
	v_cndmask_b32_e64 v13, v13, v2, s5
	v_cndmask_b32_e64 v12, v12, v2, s6
	v_cndmask_b32_e64 v11, v11, v2, s7
	v_cndmask_b32_e32 v10, v10, v1, vcc_lo
	v_cndmask_b32_e64 v9, v9, v1, s1
	v_cndmask_b32_e64 v8, v8, v1, s2
	v_cndmask_b32_e64 v7, v7, v1, s3
	v_cndmask_b32_e64 v6, v6, v1, s4
	v_cndmask_b32_e64 v5, v5, v1, s5
	v_cndmask_b32_e64 v4, v4, v1, s6
	v_cndmask_b32_e64 v3, v3, v1, s7
	s_branch .LBB0_2
.LBB0_298:
	v_dual_mov_b32 v11, s36 :: v_dual_mov_b32 v12, s37
	v_dual_mov_b32 v3, s16 :: v_dual_mov_b32 v4, s17
	v_dual_mov_b32 v13, s38 :: v_dual_mov_b32 v14, s39
	v_dual_mov_b32 v15, s40 :: v_dual_mov_b32 v16, s41
	v_dual_mov_b32 v17, s42 :: v_dual_mov_b32 v18, s43
	v_dual_mov_b32 v5, s18 :: v_dual_mov_b32 v6, s19
	v_dual_mov_b32 v7, s20 :: v_dual_mov_b32 v8, s21
	v_dual_mov_b32 v9, s22 :: v_dual_mov_b32 v10, s23
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0xff800000
.LBB0_299:                              ; %Flow1688
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_301
; %bb.300:
	s_ashr_i32 s35, s34, 31
	v_dual_mov_b32 v2, v3 :: v_dual_mov_b32 v3, v12
	s_lshl_b64 s[2:3], s[34:35], 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v1, v11 :: v_dual_mov_b32 v20, s3
	v_or_b32_e32 v19, s2, v0
	v_lshlrev_b64 v[21:22], 6, v[19:20]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v11, vcc_lo, s28, v21
	v_add_co_ci_u32_e64 v12, null, s29, v22, vcc_lo
	global_store_b128 v[11:12], v[1:4], off
	v_lshlrev_b64 v[1:2], 3, v[19:20]
	v_mov_b32_e32 v3, v13
	v_dual_mov_b32 v4, v5 :: v_dual_mov_b32 v5, v14
	v_mov_b32_e32 v19, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s28, v1
	v_add_co_ci_u32_e64 v2, null, s29, v2, vcc_lo
	global_store_b128 v[11:12], v[3:6], off offset:16
	v_mov_b32_e32 v5, v15
	v_dual_mov_b32 v6, v7 :: v_dual_mov_b32 v7, v16
	v_add_co_u32 v1, vcc_lo, 0x2a0000, v1
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v17, v9
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_clause 0x2
	global_store_b128 v[11:12], v[5:8], off offset:32
	global_store_b128 v[11:12], v[16:19], off offset:48
	global_store_b64 v[1:2], v[33:34], off
.LBB0_301:
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e64 s0, 0, v0
	s_mov_b32 s2, -1
	s_waitcnt_vscnt null, 0x0
	buffer_gl1_inv
	buffer_gl0_inv
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_324
; %bb.302:
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_304
; %bb.303:
	v_mov_b32_e32 v1, 0
	global_load_b32 v1, v1, s[30:31] offset:4 glc
.LBB0_304:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s3, exec_lo
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v1
	v_mbcnt_lo_u32_b32 v2, s3, 0
	s_mov_b32 s4, exec_lo
	buffer_gl1_inv
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v2
	s_cbranch_execz .LBB0_306
; %bb.305:
	s_ashr_i32 s6, s34, 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_u32 s6, s30, s6
	s_addc_u32 s7, s31, s7
	s_bcnt1_i32_b32 s3, s3
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v2, s3
	global_atomic_add_u32 v1, v2, s[6:7] offset:8
.LBB0_306:
	s_or_b32 exec_lo, exec_lo, s4
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .LBB0_312
; %bb.307:
	v_mov_b32_e32 v1, 0
	s_branch .LBB0_309
.LBB0_308:                              ;   in Loop: Header=BB0_309 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v2
	s_cmp_lg_u32 s3, s2
	s_cbranch_scc1 .LBB0_311
.LBB0_309:                              ; %.preheader517
                                        ; =>This Inner Loop Header: Depth=1
	v_mbcnt_lo_u32_b32 v2, exec_lo, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_308
; %bb.310:                              ;   in Loop: Header=BB0_309 Depth=1
	global_load_b32 v2, v1, s[30:31] offset:4 glc
	s_branch .LBB0_308
.LBB0_311:                              ; %Flow1598
	s_mov_b32 s2, 0
	s_branch .LBB0_323
.LBB0_312:
	s_mov_b32 s2, 0
	s_cbranch_execz .LBB0_323
; %bb.313:
	s_mov_b32 s3, exec_lo
	s_mov_b32 s2, exec_lo
	v_mbcnt_lo_u32_b32 v1, s3, 0
                                        ; implicit-def: $vgpr2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v1
	s_cbranch_execz .LBB0_315
; %bb.314:
	s_bcnt1_i32_b32 s3, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v3, s3
	global_atomic_add_u32 v2, v2, v3, s[30:31] glc
.LBB0_315:
	s_or_b32 exec_lo, exec_lo, s2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v2
	s_add_i32 s3, s33, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s6, s3, 5
	s_cmp_lt_i32 s6, 1
	v_add3_u32 v1, s2, v1, 1
	s_cbranch_scc1 .LBB0_322
; %bb.316:                              ; %.lr.ph560
	v_mov_b32_e32 v2, 0
	s_mov_b32 s3, 0
	s_add_i32 s7, s6, -1
	s_mov_b32 s2, s3
	s_set_inst_prefetch_distance 0x1
	s_branch .LBB0_318
	.p2align	6
.LBB0_317:                              ;   in Loop: Header=BB0_318 Depth=1
	s_or_b32 exec_lo, exec_lo, s12
	s_add_i32 s2, s2, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s2, s6
	s_cbranch_scc1 .LBB0_322
.LBB0_318:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_320 Depth 2
	s_lshl_b32 s4, s2, 5
	s_mov_b32 s12, 0
	s_sub_i32 s4, s33, s4
	s_cmp_eq_u32 s2, s7
	s_cselect_b32 s4, s4, 32
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_mul_lo_u32 v3, s4, v1
	s_lshl_b64 s[4:5], s[2:3], 2
	s_add_u32 s4, s30, s4
	s_addc_u32 s5, s31, s5
	s_branch .LBB0_320
.LBB0_319:                              ;   in Loop: Header=BB0_320 Depth=2
	s_or_b32 exec_lo, exec_lo, s13
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s13, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ge_i32_e32 vcc_lo, s13, v3
	s_or_b32 s12, vcc_lo, s12
	s_and_not1_b32 exec_lo, exec_lo, s12
	s_cbranch_execz .LBB0_317
.LBB0_320:                              ;   Parent Loop BB0_318 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mbcnt_lo_u32_b32 v4, exec_lo, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 0, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b32 s13, vcc_lo
	s_cbranch_execz .LBB0_319
; %bb.321:                              ;   in Loop: Header=BB0_320 Depth=2
	global_load_b32 v4, v2, s[4:5] offset:8 glc
	s_branch .LBB0_319
.LBB0_322:                              ; %._crit_edge561
	s_set_inst_prefetch_distance 0x2
	v_mov_b32_e32 v2, 0
	s_mov_b32 s2, -1
	s_waitcnt_vscnt null, 0x0
	buffer_gl1_inv
	buffer_gl0_inv
	global_store_b32 v2, v1, s[30:31] offset:4
.LBB0_323:                              ; %Flow1603
	s_or_not1_b32 s2, s2, exec_lo
.LBB0_324:                              ; %Flow1601
	s_or_b32 exec_lo, exec_lo, s1
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB0_326
; %bb.325:                              ; %.loopexit.sink.split
	s_waitcnt_vscnt null, 0x0
	buffer_gl1_inv
	buffer_gl0_inv
.LBB0_326:                              ; %.loopexit
	s_or_b32 exec_lo, exec_lo, s1
	s_cmp_ge_i32 s34, s14
	; wave barrier
	s_cbranch_scc1 .LBB0_525
; %bb.327:                              ; %.preheader516.preheader
	v_mov_b32_e32 v27, 0xff800000
	v_mov_b32_e32 v19, 0
	s_mov_b32 s13, 0
	s_mov_b32 s12, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mov_b32_e32 v28, v27
	v_mov_b32_e32 v20, v19
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v29, v27
	v_mov_b32_e32 v30, v27
	v_mov_b32_e32 v32, v27
	v_mov_b32_e32 v33, v27
	v_mov_b32_e32 v31, v27
	v_mov_b32_e32 v34, v27
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_mov_b32_e32 v24, v19
	v_mov_b32_e32 v25, v19
	v_dual_mov_b32 v11, v27 :: v_dual_mov_b32 v12, v28
	v_dual_mov_b32 v13, v29 :: v_dual_mov_b32 v14, v30
	v_dual_mov_b32 v16, v32 :: v_dual_mov_b32 v17, v33
	v_dual_mov_b32 v3, v19 :: v_dual_mov_b32 v4, v20
	v_dual_mov_b32 v5, v21 :: v_dual_mov_b32 v6, v22
	v_dual_mov_b32 v7, v23 :: v_dual_mov_b32 v8, v24
	v_mov_b32_e32 v9, v25
	v_dual_mov_b32 v15, v31 :: v_dual_mov_b32 v10, v26
	v_mov_b32_e32 v18, v34
	v_cmpx_gt_i32_e64 s33, v0
	s_cbranch_execz .LBB0_457
; %bb.328:                              ; %.preheader515.lr.ph
	s_add_u32 s14, s28, 0x2a0000
	s_addc_u32 s24, s29, 0
	s_cmp_gt_i32 s15, 0
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v27, 0xff800000
	s_cselect_b32 s25, -1, 0
	s_add_i32 s26, s15, -1
	s_ashr_i32 s27, s34, 31
	s_cmp_gt_u32 s15, 7
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v31, v27
	s_cselect_b32 s30, -1, 0
	s_cmp_gt_u32 s15, 6
	v_dual_mov_b32 v34, v27 :: v_dual_mov_b32 v9, v2
	s_cselect_b32 s31, -1, 0
	s_cmp_gt_u32 s15, 5
	v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v28, v27
	s_cselect_b32 s35, -1, 0
	s_cmp_gt_u32 s15, 4
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v29, v27
	s_cselect_b32 s36, -1, 0
	s_cmp_gt_u32 s15, 3
	v_dual_mov_b32 v5, v2 :: v_dual_mov_b32 v30, v27
	s_cselect_b32 s37, -1, 0
	s_cmp_gt_u32 s15, 2
	v_dual_mov_b32 v7, v2 :: v_dual_mov_b32 v32, v27
	s_cselect_b32 s38, -1, 0
	s_cmp_lg_u32 s15, 1
	v_dual_mov_b32 v8, v2 :: v_dual_mov_b32 v33, v27
	s_cselect_b32 s39, -1, 0
	s_cmp_lg_u32 s15, 2
	v_dual_mov_b32 v11, v27 :: v_dual_mov_b32 v16, v32
	s_cselect_b32 s40, -1, 0
	s_cmp_lg_u32 s15, 3
	v_dual_mov_b32 v15, v31 :: v_dual_mov_b32 v10, v9
	s_cselect_b32 s41, -1, 0
	s_cmp_lg_u32 s15, 4
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v14, v30
	s_cselect_b32 s42, -1, 0
	s_cmp_lg_u32 s15, 5
	v_dual_mov_b32 v19, v2 :: v_dual_mov_b32 v12, v28
	s_cselect_b32 s43, -1, 0
	s_cmp_lg_u32 s15, 6
	v_dual_mov_b32 v13, v29 :: v_dual_mov_b32 v18, v34
	s_cselect_b32 s44, -1, 0
	s_cmp_lg_u32 s15, 7
	v_mov_b32_e32 v9, v8
	v_dual_mov_b32 v17, v33 :: v_dual_mov_b32 v8, v7
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v3, v2
	s_cselect_b32 s45, -1, 0
	s_add_i32 s46, s34, 16
	s_add_i32 s48, s15, -2
	s_ashr_i32 s47, s46, 31
	s_branch .LBB0_330
.LBB0_329:                              ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_add_nc_u32_e32 v1, 32, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_le_i32_e32 vcc_lo, s33, v1
	s_or_b32 s13, vcc_lo, s13
	s_and_not1_b32 exec_lo, exec_lo, s13
	s_cbranch_execz .LBB0_456
.LBB0_330:                              ; %.preheader515
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_334 Depth 2
                                        ;     Child Loop BB0_341 Depth 2
                                        ;     Child Loop BB0_348 Depth 2
                                        ;     Child Loop BB0_355 Depth 2
                                        ;     Child Loop BB0_362 Depth 2
                                        ;     Child Loop BB0_369 Depth 2
                                        ;     Child Loop BB0_376 Depth 2
                                        ;     Child Loop BB0_383 Depth 2
                                        ;     Child Loop BB0_397 Depth 2
                                        ;     Child Loop BB0_404 Depth 2
                                        ;     Child Loop BB0_411 Depth 2
                                        ;     Child Loop BB0_418 Depth 2
                                        ;     Child Loop BB0_425 Depth 2
                                        ;     Child Loop BB0_432 Depth 2
                                        ;     Child Loop BB0_439 Depth 2
                                        ;     Child Loop BB0_446 Depth 2
	v_lshlrev_b64 v[20:21], 5, v[1:2]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v22, vcc_lo, v20, s34
	v_add_co_ci_u32_e64 v23, null, s27, v21, vcc_lo
	s_and_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccz .LBB0_387
; %bb.331:                              ; %.lr.ph581.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	v_lshlrev_b64 v[24:25], 6, v[22:23]
	s_mov_b32 s49, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s28, v24
	v_add_co_ci_u32_e64 v25, null, s29, v25, vcc_lo
	global_load_b64 v[28:29], v[24:25], off
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v28, v15
	v_cmp_gt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_and_b32 s2, s37, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, v26, 4, s1
	v_cmp_gt_f32_e64 s1, v28, v13
	v_cndmask_b32_e64 v26, v26, 3, s2
	s_and_b32 s1, s38, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 2, s1
	s_and_b32 s1, s39, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 1, s1
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_337
; %bb.332:                              ; %.preheader514
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_i32_e64 s26, v26
	s_cbranch_execz .LBB0_336
; %bb.333:                              ; %.lr.ph572.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s48
.LBB0_334:                              ; %.lr.ph572
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_mov_b32 m0, s3
	s_add_i32 s4, s3, 1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s4
	v_cmp_le_u32_e32 vcc_lo, s3, v26
	v_movreld_b32_e32 v11, v30
	s_mov_b32 m0, s3
	s_add_i32 s3, s3, -1
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s4
	s_or_b32 s2, vcc_lo, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_334
; %bb.335:                              ; %Flow1591
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_336:                              ; %Flow1592
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_337:                              ; %Flow1593
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_387
; %bb.338:                              ; %.lr.ph581.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:8
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v28, v15
	v_cmp_gt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_and_b32 s2, s37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, v26, 4, s1
	v_cmp_gt_f32_e64 s1, v28, v13
	v_cndmask_b32_e64 v26, v26, 3, s2
	s_and_b32 s1, s38, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 2, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_344
; %bb.339:                              ; %.preheader514.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_343
; %bb.340:                              ; %.lr.ph572.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_341:                              ; %.lr.ph572.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_341
; %bb.342:                              ; %Flow1587
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_343:                              ; %Flow1588
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_344:                              ; %Flow1589
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_387
; %bb.345:                              ; %.lr.ph581.2
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:16
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v28, v15
	v_cmp_gt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 4, s1
	s_and_b32 s1, s37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 3, s1
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_351
; %bb.346:                              ; %.preheader514.2
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_350
; %bb.347:                              ; %.lr.ph572.2.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_348:                              ; %.lr.ph572.2
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_348
; %bb.349:                              ; %Flow1583
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_350:                              ; %Flow1584
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_351:                              ; %Flow1585
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_387
; %bb.352:                              ; %.lr.ph581.3
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:24
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v28, v15
	v_cmp_ngt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 4, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v13
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_358
; %bb.353:                              ; %.preheader514.3
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_357
; %bb.354:                              ; %.lr.ph572.3.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_355:                              ; %.lr.ph572.3
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_355
; %bb.356:                              ; %Flow1579
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_357:                              ; %Flow1580
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_358:                              ; %Flow1581
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s42
	s_cbranch_vccnz .LBB0_387
; %bb.359:                              ; %.lr.ph581.4
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:32
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 6, s1
	s_and_b32 s1, s35, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 5, s1
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v13
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_365
; %bb.360:                              ; %.preheader514.4
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_364
; %bb.361:                              ; %.lr.ph572.4.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_362:                              ; %.lr.ph572.4
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_362
; %bb.363:                              ; %Flow1575
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_364:                              ; %Flow1576
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_365:                              ; %Flow1577
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_cbranch_vccnz .LBB0_387
; %bb.366:                              ; %.lr.ph581.5
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:40
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	v_cmp_gt_f32_e64 s1, v28, v17
	s_and_b32 s2, s30, vcc_lo
	s_and_b32 s1, s31, s1
	v_cndmask_b32_e64 v26, s15, 7, s2
	v_cmp_ngt_f32_e32 vcc_lo, v28, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cndmask_b32_e32 v26, 5, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_372
; %bb.367:                              ; %.preheader514.5
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_371
; %bb.368:                              ; %.lr.ph572.5.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_369:                              ; %.lr.ph572.5
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_369
; %bb.370:                              ; %Flow1571
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_371:                              ; %Flow1572
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_372:                              ; %Flow1573
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s44
	s_cbranch_vccnz .LBB0_387
; %bb.373:                              ; %.lr.ph581.6
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[28:29], v[24:25], off offset:48
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v28, v18
	s_and_b32 s1, s30, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v17
	v_cndmask_b32_e64 v26, s15, 7, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 6, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v16
	v_cndmask_b32_e32 v26, 5, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v14
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v12
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v28, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_379
; %bb.374:                              ; %.preheader514.6
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_378
; %bb.375:                              ; %.lr.ph572.6.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_376:                              ; %.lr.ph572.6
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v30, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v30
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v30, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v30
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_376
; %bb.377:                              ; %Flow1567
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_378:                              ; %Flow1568
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v28, vcc_lo
	v_cndmask_b32_e64 v17, v17, v28, s1
	v_cndmask_b32_e64 v16, v16, v28, s2
	v_cndmask_b32_e64 v15, v15, v28, s3
	v_cndmask_b32_e64 v14, v14, v28, s4
	v_cndmask_b32_e64 v13, v13, v28, s5
	v_cndmask_b32_e64 v12, v12, v28, s6
	v_cndmask_b32_e64 v11, v11, v28, s7
	v_cndmask_b32_e32 v10, v10, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v29, s1
	v_cndmask_b32_e64 v8, v8, v29, s2
	v_cndmask_b32_e64 v7, v7, v29, s3
	v_cndmask_b32_e64 v6, v6, v29, s4
	v_cndmask_b32_e64 v5, v5, v29, s5
	v_cndmask_b32_e64 v4, v4, v29, s6
	v_cndmask_b32_e64 v3, v3, v29, s7
.LBB0_379:                              ; %Flow1569
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s45
	s_cbranch_vccnz .LBB0_387
; %bb.380:                              ; %.lr.ph581.7
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[24:25], off offset:56
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_ngt_f32_e32 vcc_lo, v24, v18
	v_cndmask_b32_e64 v26, 7, s15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 6, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e32 v26, 5, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_386
; %bb.381:                              ; %.preheader514.7
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_385
; %bb.382:                              ; %.lr.ph572.7.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_383:                              ; %.lr.ph572.7
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_383
; %bb.384:                              ; %Flow1563
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_385:                              ; %Flow1564
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_386:                              ; %Flow1565
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
.LBB0_387:                              ; %._crit_edge582
                                        ;   in Loop: Header=BB0_330 Depth=1
	v_lshlrev_b64 v[22:23], 3, v[22:23]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v22, vcc_lo, s14, v22
	v_add_co_ci_u32_e64 v23, null, s24, v23, vcc_lo
	global_load_b64 v[22:23], v[22:23], off
	s_waitcnt vmcnt(0)
	v_cmpx_ngt_f32_e32 v22, v27
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_391
; %bb.388:                              ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_lg_f32_e32 0xff800000, v22
	s_cbranch_execz .LBB0_390
; %bb.389:                              ;   in Loop: Header=BB0_330 Depth=1
	v_sub_f32_e32 v22, v22, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v24, 0x3fb8aa3b, v22
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v22
	v_fma_f32 v25, 0x3fb8aa3b, v22, -v24
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, 0x32a5705f, v22
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v24, v24, v25
	v_cvt_i32_f32_e32 v25, v26
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v22
	v_cndmask_b32_e32 v22, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, v23, v22
.LBB0_390:                              ; %Flow1561
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
                                        ; implicit-def: $vgpr22_vgpr23
.LBB0_391:                              ; %Flow1562
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_393
; %bb.392:                              ;   in Loop: Header=BB0_330 Depth=1
	v_sub_f32_e32 v24, v27, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v25, 0x3fb8aa3b, v24
	v_fma_f32 v26, 0x3fb8aa3b, v24, -v25
	v_rndne_f32_e32 v27, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v25, v25, v27
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v24
	v_fmac_f32_e32 v26, 0x32a5705f, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v25, v25, v26
	v_cvt_i32_f32_e32 v26, v27
	v_mov_b32_e32 v27, v22
	v_exp_f32_e32 v25, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v25, v25, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v25, 0, v25, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v24
	v_cndmask_b32_e32 v24, 0x7f800000, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v19, v24
	v_mov_b32_e32 v19, v23
.LBB0_393:                              ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s1
	v_add_co_u32 v20, vcc_lo, v20, s46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, null, s47, v21, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_cbranch_vccnz .LBB0_450
; %bb.394:                              ; %.lr.ph581.preheader.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	v_lshlrev_b64 v[22:23], 6, v[20:21]
	s_mov_b32 s49, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v22, vcc_lo, s28, v22
	v_add_co_ci_u32_e64 v23, null, s29, v23, vcc_lo
	global_load_b64 v[24:25], v[22:23], off
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v24, v15
	v_cmp_gt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_and_b32 s2, s37, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, v26, 4, s1
	v_cmp_gt_f32_e64 s1, v24, v13
	v_cndmask_b32_e64 v26, v26, 3, s2
	s_and_b32 s1, s38, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 2, s1
	s_and_b32 s1, s39, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 1, s1
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_400
; %bb.395:                              ; %.preheader514.1687
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_399
; %bb.396:                              ; %.lr.ph572.1694.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_397:                              ; %.lr.ph572.1694
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_397
; %bb.398:                              ; %Flow1557
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_399:                              ; %Flow1558
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_400:                              ; %Flow1559
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_cbranch_vccnz .LBB0_450
; %bb.401:                              ; %.lr.ph581.1.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:8
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v24, v15
	v_cmp_gt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_and_b32 s2, s37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, v26, 4, s1
	v_cmp_gt_f32_e64 s1, v24, v13
	v_cndmask_b32_e64 v26, v26, 3, s2
	s_and_b32 s1, s38, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 2, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_407
; %bb.402:                              ; %.preheader514.1.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_406
; %bb.403:                              ; %.lr.ph572.1.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_404:                              ; %.lr.ph572.1.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_404
; %bb.405:                              ; %Flow1553
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_406:                              ; %Flow1554
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_407:                              ; %Flow1555
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s40
	s_cbranch_vccnz .LBB0_450
; %bb.408:                              ; %.lr.ph581.2.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:16
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v24, v15
	v_cmp_gt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 4, s1
	s_and_b32 s1, s37, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 3, s1
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_414
; %bb.409:                              ; %.preheader514.2.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_413
; %bb.410:                              ; %.lr.ph572.2.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_411:                              ; %.lr.ph572.2.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_411
; %bb.412:                              ; %Flow1549
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_413:                              ; %Flow1550
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_414:                              ; %Flow1551
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_cbranch_vccnz .LBB0_450
; %bb.415:                              ; %.lr.ph581.3.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:24
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_and_b32 s2, s35, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cmp_gt_f32_e64 s1, v24, v15
	v_cmp_ngt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e64 v26, v26, 5, s2
	s_and_b32 s1, s36, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 4, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_421
; %bb.416:                              ; %.preheader514.3.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_420
; %bb.417:                              ; %.lr.ph572.3.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_418:                              ; %.lr.ph572.3.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_418
; %bb.419:                              ; %Flow1545
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_420:                              ; %Flow1546
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_421:                              ; %Flow1547
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s42
	s_cbranch_vccnz .LBB0_450
; %bb.422:                              ; %.lr.ph581.4.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:32
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e64 v26, s15, 7, s2
	s_and_b32 s1, s31, s1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cndmask_b32_e64 v26, v26, 6, s1
	s_and_b32 s1, s35, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 5, s1
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_428
; %bb.423:                              ; %.preheader514.4.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_427
; %bb.424:                              ; %.lr.ph572.4.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_425:                              ; %.lr.ph572.4.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_425
; %bb.426:                              ; %Flow1541
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_427:                              ; %Flow1542
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_428:                              ; %Flow1543
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_cbranch_vccnz .LBB0_450
; %bb.429:                              ; %.lr.ph581.5.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:40
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	v_cmp_gt_f32_e64 s1, v24, v17
	s_and_b32 s2, s30, vcc_lo
	s_and_b32 s1, s31, s1
	v_cndmask_b32_e64 v26, s15, 7, s2
	v_cmp_ngt_f32_e32 vcc_lo, v24, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v26, v26, 6, s1
	v_cndmask_b32_e32 v26, 5, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_435
; %bb.430:                              ; %.preheader514.5.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_434
; %bb.431:                              ; %.lr.ph572.5.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_432:                              ; %.lr.ph572.5.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_432
; %bb.433:                              ; %Flow1537
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_434:                              ; %Flow1538
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_435:                              ; %Flow1539
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s44
	s_cbranch_vccnz .LBB0_450
; %bb.436:                              ; %.lr.ph581.6.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[24:25], v[22:23], off offset:48
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v24, v18
	s_and_b32 s1, s30, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v17
	v_cndmask_b32_e64 v26, s15, 7, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 6, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v16
	v_cndmask_b32_e32 v26, 5, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 4, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v14
	v_cndmask_b32_e32 v26, 3, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v26, 2, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v12
	v_cndmask_b32_e32 v26, 1, v26, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v24, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v26, 0, v26, vcc_lo
	v_cmpx_gt_u32_e64 s15, v26
	s_cbranch_execz .LBB0_442
; %bb.437:                              ; %.preheader514.6.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v26
	s_cbranch_execz .LBB0_441
; %bb.438:                              ; %.lr.ph572.6.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_439:                              ; %.lr.ph572.6.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v28, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v26
	v_movreld_b32_e32 v11, v28
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v28, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v28
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_439
; %bb.440:                              ; %Flow1533
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_441:                              ; %Flow1534
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v26
	v_cmp_eq_u32_e64 s1, 6, v26
	v_cmp_eq_u32_e64 s2, 5, v26
	v_cmp_eq_u32_e64 s3, 4, v26
	v_cmp_eq_u32_e64 s4, 3, v26
	v_cmp_eq_u32_e64 s5, 2, v26
	v_cmp_eq_u32_e64 s6, 1, v26
	v_cmp_eq_u32_e64 s7, 0, v26
	v_cndmask_b32_e32 v18, v18, v24, vcc_lo
	v_cndmask_b32_e64 v17, v17, v24, s1
	v_cndmask_b32_e64 v16, v16, v24, s2
	v_cndmask_b32_e64 v15, v15, v24, s3
	v_cndmask_b32_e64 v14, v14, v24, s4
	v_cndmask_b32_e64 v13, v13, v24, s5
	v_cndmask_b32_e64 v12, v12, v24, s6
	v_cndmask_b32_e64 v11, v11, v24, s7
	v_cndmask_b32_e32 v10, v10, v25, vcc_lo
	v_cndmask_b32_e64 v9, v9, v25, s1
	v_cndmask_b32_e64 v8, v8, v25, s2
	v_cndmask_b32_e64 v7, v7, v25, s3
	v_cndmask_b32_e64 v6, v6, v25, s4
	v_cndmask_b32_e64 v5, v5, v25, s5
	v_cndmask_b32_e64 v4, v4, v25, s6
	v_cndmask_b32_e64 v3, v3, v25, s7
.LBB0_442:                              ; %Flow1535
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s45
	s_cbranch_vccnz .LBB0_450
; %bb.443:                              ; %.lr.ph581.7.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	global_load_b64 v[22:23], v[22:23], off offset:56
	s_mov_b32 s49, exec_lo
	s_waitcnt vmcnt(0)
	v_cmp_ngt_f32_e32 vcc_lo, v22, v18
	v_cndmask_b32_e64 v24, 7, s15, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 6, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v16
	v_cndmask_b32_e32 v24, 5, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 4, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v14
	v_cndmask_b32_e32 v24, 3, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v24, 2, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v12
	v_cndmask_b32_e32 v24, 1, v24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v22, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmpx_gt_u32_e64 s15, v24
	s_cbranch_execz .LBB0_449
; %bb.444:                              ; %.preheader514.7.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s1, exec_lo
	v_cmpx_gt_u32_e64 s26, v24
	s_cbranch_execz .LBB0_448
; %bb.445:                              ; %.lr.ph572.7.1.preheader
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, 0
	s_mov_b32 s3, s26
.LBB0_446:                              ; %.lr.ph572.7.1
                                        ;   Parent Loop BB0_330 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	s_add_i32 m0, s3, -1
	s_add_i32 s4, s3, -1
	v_movrels_b32_e32 v25, v11
	s_mov_b32 m0, s3
	v_cmp_le_u32_e32 vcc_lo, s4, v24
	v_movreld_b32_e32 v11, v25
	s_add_i32 m0, s3, -1
	s_or_b32 s2, vcc_lo, s2
	v_movrels_b32_e32 v25, v3
	s_mov_b32 m0, s3
	s_mov_b32 s3, s4
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v3, v25
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_446
; %bb.447:                              ; %Flow1529
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
.LBB0_448:                              ; %Flow1530
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s1
	v_cmp_eq_u32_e32 vcc_lo, 7, v24
	v_cmp_eq_u32_e64 s1, 6, v24
	v_cmp_eq_u32_e64 s2, 5, v24
	v_cmp_eq_u32_e64 s3, 4, v24
	v_cmp_eq_u32_e64 s4, 3, v24
	v_cmp_eq_u32_e64 s5, 2, v24
	v_cmp_eq_u32_e64 s6, 1, v24
	v_cmp_eq_u32_e64 s7, 0, v24
	v_cndmask_b32_e32 v18, v18, v22, vcc_lo
	v_cndmask_b32_e64 v17, v17, v22, s1
	v_cndmask_b32_e64 v16, v16, v22, s2
	v_cndmask_b32_e64 v15, v15, v22, s3
	v_cndmask_b32_e64 v14, v14, v22, s4
	v_cndmask_b32_e64 v13, v13, v22, s5
	v_cndmask_b32_e64 v12, v12, v22, s6
	v_cndmask_b32_e64 v11, v11, v22, s7
	v_cndmask_b32_e32 v10, v10, v23, vcc_lo
	v_cndmask_b32_e64 v9, v9, v23, s1
	v_cndmask_b32_e64 v8, v8, v23, s2
	v_cndmask_b32_e64 v7, v7, v23, s3
	v_cndmask_b32_e64 v6, v6, v23, s4
	v_cndmask_b32_e64 v5, v5, v23, s5
	v_cndmask_b32_e64 v4, v4, v23, s6
	v_cndmask_b32_e64 v3, v3, v23, s7
.LBB0_449:                              ; %Flow1531
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s49
.LBB0_450:                              ; %._crit_edge582.1
                                        ;   in Loop: Header=BB0_330 Depth=1
	v_lshlrev_b64 v[20:21], 3, v[20:21]
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v20, vcc_lo, s14, v20
	v_add_co_ci_u32_e64 v21, null, s24, v21, vcc_lo
	global_load_b64 v[20:21], v[20:21], off
	s_waitcnt vmcnt(0)
	v_cmpx_ngt_f32_e32 v20, v27
	s_xor_b32 s1, exec_lo, s1
	s_cbranch_execz .LBB0_454
; %bb.451:                              ;   in Loop: Header=BB0_330 Depth=1
	s_mov_b32 s2, exec_lo
	v_cmpx_lg_f32_e32 0xff800000, v20
	s_cbranch_execz .LBB0_453
; %bb.452:                              ;   in Loop: Header=BB0_330 Depth=1
	v_sub_f32_e32 v20, v20, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v22, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v23, 0x3fb8aa3b, v20, -v22
	v_rndne_f32_e32 v24, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v23, 0x32a5705f, v20
	v_sub_f32_e32 v22, v22, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v22, v22, v23
	v_cvt_i32_f32_e32 v23, v24
	v_exp_f32_e32 v22, v22
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v22, 0, v22, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v22, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, v21, v20
.LBB0_453:                              ; %Flow1527
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
                                        ; implicit-def: $vgpr20_vgpr21
.LBB0_454:                              ; %Flow1528
                                        ;   in Loop: Header=BB0_330 Depth=1
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB0_329
; %bb.455:                              ;   in Loop: Header=BB0_330 Depth=1
	v_dual_sub_f32 v22, v27, v20 :: v_dual_mov_b32 v27, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v23, 0x3fb8aa3b, v22
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v22
	v_fma_f32 v24, 0x3fb8aa3b, v22, -v23
	v_rndne_f32_e32 v25, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v24, 0x32a5705f, v22 :: v_dual_sub_f32 v23, v23, v25
	v_add_f32_e32 v23, v23, v24
	v_cvt_i32_f32_e32 v24, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v23, v23
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v23, v23, v24
	v_cndmask_b32_e32 v23, 0, v23, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v22, 0x7f800000, v23, vcc_lo
	v_fmac_f32_e32 v21, v19, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v19, v21
	s_branch .LBB0_329
.LBB0_456:                              ; %Flow1595
	s_or_b32 exec_lo, exec_lo, s13
.LBB0_457:                              ; %Flow1596
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s12
	v_dual_mov_b32 v1, v11 :: v_dual_lshlrev_b32 v20, 6, v0
	v_dual_mov_b32 v2, v3 :: v_dual_mov_b32 v3, v12
	v_lshlrev_b32_e32 v0, 3, v0
	v_dual_mov_b32 v28, v19 :: v_dual_mov_b32 v19, v10
	ds_store_b128 v20, v[1:4]
	v_mov_b32_e32 v3, v13
	v_dual_mov_b32 v4, v5 :: v_dual_mov_b32 v5, v14
	ds_store_b128 v20, v[3:6] offset:16
	v_mov_b32_e32 v5, v15
	v_dual_mov_b32 v6, v7 :: v_dual_mov_b32 v7, v16
	v_mov_b32_e32 v16, v17
	v_mov_b32_e32 v17, v9
	ds_store_b128 v20, v[5:8] offset:32
	ds_store_b128 v20, v[16:19] offset:48
	ds_store_b64 v0, v[27:28] offset:2048
	; wave barrier
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_525
; %bb.458:                              ; %.preheader512
	s_cmp_gt_i32 s15, 0
	s_mov_b32 s36, 0xff800000
	s_cselect_b32 s6, -1, 0
	s_add_i32 s7, s15, -1
	s_cmp_gt_u32 s15, 7
	s_mov_b32 s37, s36
	s_cselect_b32 s12, -1, 0
	s_cmp_gt_u32 s15, 6
	s_mov_b32 s38, s36
	s_cselect_b32 s13, -1, 0
	s_cmp_gt_u32 s15, 5
	s_mov_b32 s39, s36
	s_cselect_b32 s14, -1, 0
	s_cmp_gt_u32 s15, 4
	s_mov_b32 s40, s36
	s_cselect_b32 s24, -1, 0
	s_cmp_gt_u32 s15, 3
	s_mov_b32 s41, s36
	s_cselect_b32 s25, -1, 0
	s_cmp_gt_u32 s15, 2
	s_mov_b32 s42, s36
	s_cselect_b32 s26, -1, 0
	s_cmp_lg_u32 s15, 1
	s_mov_b32 s43, s36
	s_cselect_b32 s27, -1, 0
	s_cmp_lg_u32 s15, 2
	v_dual_mov_b32 v0, s16 :: v_dual_mov_b32 v1, s17
	s_cselect_b32 s28, -1, 0
	s_cmp_lg_u32 s15, 3
	v_dual_mov_b32 v8, s36 :: v_dual_mov_b32 v9, s37
	s_cselect_b32 s29, -1, 0
	s_cmp_lg_u32 s15, 4
	v_dual_mov_b32 v2, s18 :: v_dual_mov_b32 v3, s19
	s_cselect_b32 s30, -1, 0
	s_cmp_lg_u32 s15, 5
	v_dual_mov_b32 v4, s20 :: v_dual_mov_b32 v5, s21
	s_cselect_b32 s31, -1, 0
	s_cmp_lg_u32 s15, 6
	v_dual_mov_b32 v6, s22 :: v_dual_mov_b32 v7, s23
	v_dual_mov_b32 v10, s38 :: v_dual_mov_b32 v11, s39
	v_dual_mov_b32 v12, s40 :: v_dual_mov_b32 v13, s41
	v_dual_mov_b32 v14, s42 :: v_dual_mov_b32 v15, s43
	v_cndmask_b32_e64 v18, 0, 1, s6
	s_cselect_b32 s33, -1, 0
	s_cmp_lg_u32 s15, 7
	s_mov_b32 s16, 0
	s_cselect_b32 s17, -1, 0
	s_branch .LBB0_461
.LBB0_459:                              ; %._crit_edge609.7
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_460:                              ; %._crit_edge616
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_add_i32 s16, s16, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s16, 32
	s_cbranch_scc0 .LBB0_508
.LBB0_461:                              ; %.preheader511
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_465 Depth 2
                                        ;     Child Loop BB0_471 Depth 2
                                        ;     Child Loop BB0_477 Depth 2
                                        ;     Child Loop BB0_483 Depth 2
                                        ;     Child Loop BB0_489 Depth 2
                                        ;     Child Loop BB0_495 Depth 2
                                        ;     Child Loop BB0_501 Depth 2
                                        ;     Child Loop BB0_507 Depth 2
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_460
; %bb.462:                              ; %.lr.ph615
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_lshl_b32 s5, s16, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	s_and_b32 s1, s12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v13
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s1, 7, s15
	s_and_b32 s2, s13, s0
	v_cmp_gt_f32_e64 s0, v16, v12
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 6, s1
	s_and_b32 s2, s14, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v11
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 5, s1
	s_and_b32 s2, s24, s0
	v_cmp_gt_f32_e64 s0, v16, v10
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 4, s1
	s_and_b32 s2, s25, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v9
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 3, s1
	s_and_b32 s2, s26, s0
	v_cmp_ngt_f32_e64 s0, v16, v8
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 2, s1
	s_and_b32 s2, s27, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 1, s1
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s1, 0
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_467
; %bb.463:                              ; %.preheader510
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_466
; %bb.464:                              ; %.lr.ph608.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_465:                              ; %.lr.ph608
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_le_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc0 .LBB0_465
.LBB0_466:                              ; %Flow1523
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_467:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_cbranch_vccnz .LBB0_460
; %bb.468:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:8
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	s_and_b32 s1, s12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v13
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s1, 7, s15
	s_and_b32 s2, s13, s0
	v_cmp_gt_f32_e64 s0, v16, v12
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 6, s1
	s_and_b32 s2, s14, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v11
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 5, s1
	s_and_b32 s2, s24, s0
	v_cmp_gt_f32_e64 s0, v16, v10
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 4, s1
	s_and_b32 s2, s25, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v9
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 3, s1
	s_and_b32 s2, s26, s0
	v_cmp_ngt_f32_e64 s0, v16, v8
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 2, s1
	s_and_b32 s2, vcc_lo, exec_lo
	s_cselect_b32 s1, s1, 1
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_473
; %bb.469:                              ; %.preheader510.1
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_472
; %bb.470:                              ; %.lr.ph608.1.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_471:                              ; %.lr.ph608.1
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_471
.LBB0_472:                              ; %._crit_edge609.1
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_473:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_cbranch_vccnz .LBB0_460
; %bb.474:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:16
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	s_and_b32 s1, s12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v13
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s1, 7, s15
	s_and_b32 s2, s13, s0
	v_cmp_gt_f32_e64 s0, v16, v12
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 6, s1
	s_and_b32 s2, s14, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v11
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 5, s1
	s_and_b32 s2, s24, s0
	v_cmp_ngt_f32_e64 s0, v16, v10
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, 4, s1
	s_and_b32 s2, s25, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v9
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s2, 3, s1
	v_cmp_ngt_f32_e64 s1, v16, v8
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s2, 2
	s_and_b32 s2, vcc_lo, exec_lo
	s_cselect_b32 s0, s0, 1
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s0, s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_479
; %bb.475:                              ; %.preheader510.2
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_478
; %bb.476:                              ; %.lr.ph608.2.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_477:                              ; %.lr.ph608.2
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_477
.LBB0_478:                              ; %._crit_edge609.2
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_479:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_cbranch_vccnz .LBB0_460
; %bb.480:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:24
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	v_cmp_gt_f32_e64 s1, v16, v13
	s_and_b32 s2, s12, vcc_lo
	v_cmp_gt_f32_e32 vcc_lo, v16, v12
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s2, 7, s15
	s_and_b32 s3, s13, s0
	v_cmp_ngt_f32_e64 s0, v16, v11
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, 6, s2
	s_and_b32 s3, s14, s1
	v_cmp_ngt_f32_e64 s1, v16, v10
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, 5, s2
	s_and_b32 s3, s24, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v9
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, 4, s2
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s2, s2, 3
	v_cmp_ngt_f32_e64 s0, v16, v8
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s1, s2, 2
	s_and_b32 s2, vcc_lo, exec_lo
	s_cselect_b32 s1, s1, 1
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_485
; %bb.481:                              ; %.preheader510.3
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_484
; %bb.482:                              ; %.lr.ph608.3.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_483:                              ; %.lr.ph608.3
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_483
.LBB0_484:                              ; %._crit_edge609.3
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_485:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_cbranch_vccnz .LBB0_460
; %bb.486:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:32
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	v_cmp_gt_f32_e64 s1, v16, v13
	v_cmp_ngt_f32_e64 s2, v16, v12
	s_and_b32 s3, s12, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v11
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s3, 7, s15
	s_and_b32 s4, s13, s0
	v_cmp_ngt_f32_e64 s0, v16, v10
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s3, 6, s3
	s_and_b32 s4, s14, s1
	v_cmp_ngt_f32_e64 s1, v16, v9
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s3, 5, s3
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s2, s3, 4
	s_and_b32 s3, vcc_lo, exec_lo
	s_cselect_b32 s2, s2, 3
	v_cmp_ngt_f32_e32 vcc_lo, v16, v8
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s2, 2
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s0, s0, 1
	s_and_b32 s1, vcc_lo, exec_lo
	s_cselect_b32 s0, s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_491
; %bb.487:                              ; %.preheader510.4
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_490
; %bb.488:                              ; %.lr.ph608.4.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_489:                              ; %.lr.ph608.4
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_489
.LBB0_490:                              ; %._crit_edge609.4
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_491:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s31
	s_cbranch_vccnz .LBB0_460
; %bb.492:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:40
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_gt_f32_e64 s0, v16, v14
	v_cmp_ngt_f32_e64 s1, v16, v13
	v_cmp_ngt_f32_e64 s2, v16, v12
	v_cmp_ngt_f32_e64 s3, v16, v11
	s_and_b32 s4, s12, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v10
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s4, 7, s15
	s_and_b32 s18, s13, s0
	v_cmp_ngt_f32_e64 s0, v16, v9
	s_and_b32 s18, s18, exec_lo
	s_cselect_b32 s4, 6, s4
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s1, s4, 5
	s_and_b32 s2, s2, exec_lo
	s_cselect_b32 s1, s1, 4
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, s1, 3
	v_cmp_ngt_f32_e64 s1, v16, v8
	s_and_b32 s3, vcc_lo, exec_lo
	s_cselect_b32 s2, s2, 2
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s2, 1
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s0, s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_497
; %bb.493:                              ; %.preheader510.5
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_496
; %bb.494:                              ; %.lr.ph608.5.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_495:                              ; %.lr.ph608.5
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_495
.LBB0_496:                              ; %._crit_edge609.5
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_497:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_cbranch_vccnz .LBB0_460
; %bb.498:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:48
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v15
	v_cmp_ngt_f32_e64 s0, v16, v14
	v_cmp_ngt_f32_e64 s1, v16, v13
	v_cmp_ngt_f32_e64 s2, v16, v12
	v_cmp_ngt_f32_e64 s3, v16, v11
	s_and_b32 s18, s12, vcc_lo
	v_cmp_ngt_f32_e64 s4, v16, v10
	s_and_b32 s18, s18, exec_lo
	s_cselect_b32 s18, 7, s15
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s18, 6
	s_and_b32 s1, s1, exec_lo
	s_cselect_b32 s0, s0, 5
	s_and_b32 s1, s2, exec_lo
	v_cmp_ngt_f32_e32 vcc_lo, v16, v9
	s_cselect_b32 s0, s0, 4
	s_and_b32 s1, s3, exec_lo
	s_cselect_b32 s1, s0, 3
	v_cmp_ngt_f32_e64 s0, v16, v8
	s_and_b32 s2, s4, exec_lo
	s_cselect_b32 s1, s1, 2
	s_and_b32 s2, vcc_lo, exec_lo
	s_cselect_b32 s1, s1, 1
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_503
; %bb.499:                              ; %.preheader510.6
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_502
; %bb.500:                              ; %.lr.ph608.6.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_501:                              ; %.lr.ph608.6
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_501
.LBB0_502:                              ; %._crit_edge609.6
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 m0, s0
	v_movreld_b32_e32 v8, v16
	v_movreld_b32_e32 v0, v17
.LBB0_503:                              ;   in Loop: Header=BB0_461 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_cbranch_vccnz .LBB0_460
; %bb.504:                              ;   in Loop: Header=BB0_461 Depth=1
	v_mov_b32_e32 v16, s5
	ds_load_b64 v[16:17], v16 offset:56
	s_waitcnt lgkmcnt(0)
	v_cmp_ngt_f32_e32 vcc_lo, v16, v15
	v_cmp_ngt_f32_e64 s0, v16, v14
	v_cmp_ngt_f32_e64 s1, v16, v13
	v_cmp_ngt_f32_e64 s2, v16, v12
	v_cmp_ngt_f32_e64 s3, v16, v11
	s_and_b32 s18, vcc_lo, exec_lo
	s_cselect_b32 s18, s15, 7
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s0, s18, 6
	s_and_b32 s1, s1, exec_lo
	v_cmp_ngt_f32_e64 s4, v16, v10
	s_cselect_b32 s0, s0, 5
	s_and_b32 s1, s2, exec_lo
	v_cmp_ngt_f32_e64 s5, v16, v9
	s_cselect_b32 s0, s0, 4
	s_and_b32 s1, s3, exec_lo
	s_cselect_b32 s0, s0, 3
	v_cmp_ngt_f32_e32 vcc_lo, v16, v8
	s_and_b32 s1, s4, exec_lo
	s_cselect_b32 s0, s0, 2
	s_and_b32 s1, s5, exec_lo
	s_cselect_b32 s0, s0, 1
	s_and_b32 s1, vcc_lo, exec_lo
	s_cselect_b32 s0, s0, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_ge_u32 s0, s15
	s_cbranch_scc1 .LBB0_460
; %bb.505:                              ; %.preheader510.7
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_cmp_le_u32 s7, s0
	s_cbranch_scc1 .LBB0_459
; %bb.506:                              ; %.lr.ph608.7.preheader
                                        ;   in Loop: Header=BB0_461 Depth=1
	s_mov_b32 s1, s7
.LBB0_507:                              ; %.lr.ph608.7
                                        ;   Parent Loop BB0_461 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_i32 m0, s1, -1
	s_add_i32 s2, s1, -1
	v_movrels_b32_e32 v19, v8
	s_mov_b32 m0, s1
	v_movreld_b32_e32 v8, v19
	s_add_i32 m0, s1, -1
	s_cmp_gt_u32 s2, s0
	v_movrels_b32_e32 v19, v0
	s_mov_b32 m0, s1
	s_mov_b32 s1, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_movreld_b32_e32 v0, v19
	s_cbranch_scc1 .LBB0_507
	s_branch .LBB0_459
.LBB0_508:                              ; %.preheader.preheader
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0xff800000
	s_movk_i32 s0, 0xff00
	s_branch .LBB0_510
.LBB0_509:                              ;   in Loop: Header=BB0_510 Depth=1
	s_add_i32 s0, s0, 16
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s0, 0
	s_cbranch_scc1 .LBB0_522
.LBB0_510:                              ; %.preheader
                                        ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v16, s0
	ds_load_b64 v[16:17], v16 offset:2304
	s_waitcnt lgkmcnt(0)
	v_cmp_ngt_f32_e32 vcc_lo, v16, v19
	s_cbranch_vccz .LBB0_514
; %bb.511:                              ;   in Loop: Header=BB0_510 Depth=1
	v_cmp_nlg_f32_e32 vcc_lo, 0xff800000, v16
	v_mov_b32_e32 v21, v20
	s_cbranch_vccnz .LBB0_513
; %bb.512:                              ;   in Loop: Header=BB0_510 Depth=1
	v_sub_f32_e32 v21, v16, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, 0x3fb8aa3b, v21
	v_fma_f32 v23, 0x3fb8aa3b, v21, -v22
	v_rndne_f32_e32 v24, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v22, v22, v24
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v21
	v_fmac_f32_e32 v23, 0x32a5705f, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v22, v22, v23
	v_cvt_i32_f32_e32 v23, v24
	v_exp_f32_e32 v22, v22
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v22, 0, v22, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v21
	v_cndmask_b32_e32 v21, 0x7f800000, v22, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v21, v17, v21, v20
.LBB0_513:                              ; %Flow1492
                                        ;   in Loop: Header=BB0_510 Depth=1
	s_cbranch_execz .LBB0_515
	s_branch .LBB0_516
.LBB0_514:                              ;   in Loop: Header=BB0_510 Depth=1
                                        ; implicit-def: $vgpr21
.LBB0_515:                              ;   in Loop: Header=BB0_510 Depth=1
	v_sub_f32_e32 v19, v19, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, 0x3fb8aa3b, v19
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	v_fma_f32 v22, 0x3fb8aa3b, v19, -v21
	v_rndne_f32_e32 v23, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v22, 0x32a5705f, v19
	v_sub_f32_e32 v21, v21, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v21, v22
	v_cvt_i32_f32_e32 v22, v23
	v_exp_f32_e32 v21, v21
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v21, v21, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v21, 0, v21, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	v_cndmask_b32_e32 v19, 0x7f800000, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v17, v20, v19
	v_mov_b32_e32 v19, v16
	v_mov_b32_e32 v21, v17
.LBB0_516:                              ; %.preheader.1
                                        ;   in Loop: Header=BB0_510 Depth=1
	v_mov_b32_e32 v16, s0
	ds_load_b64 v[16:17], v16 offset:2312
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_f32_e32 vcc_lo, v16, v19
	s_cbranch_vccnz .LBB0_520
; %bb.517:                              ;   in Loop: Header=BB0_510 Depth=1
	v_cmp_nlg_f32_e32 vcc_lo, 0xff800000, v16
	v_mov_b32_e32 v20, v21
	s_cbranch_vccnz .LBB0_519
; %bb.518:                              ;   in Loop: Header=BB0_510 Depth=1
	v_sub_f32_e32 v20, v16, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v22, 0x3fb8aa3b, v20
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v20
	v_fma_f32 v23, 0x3fb8aa3b, v20, -v22
	v_rndne_f32_e32 v24, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v23, 0x32a5705f, v20
	v_sub_f32_e32 v22, v22, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v22, v22, v23
	v_cvt_i32_f32_e32 v23, v24
	v_exp_f32_e32 v22, v22
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v22, v22, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v22, 0, v22, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v20
	v_cndmask_b32_e32 v20, 0x7f800000, v22, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v20, v17, v20, v21
.LBB0_519:                              ; %Flow1490
                                        ;   in Loop: Header=BB0_510 Depth=1
	s_cbranch_execnz .LBB0_509
	s_branch .LBB0_521
.LBB0_520:                              ;   in Loop: Header=BB0_510 Depth=1
                                        ; implicit-def: $vgpr20
.LBB0_521:                              ;   in Loop: Header=BB0_510 Depth=1
	v_sub_f32_e32 v19, v19, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v20, 0x3fb8aa3b, v19
	v_fma_f32 v22, 0x3fb8aa3b, v19, -v20
	v_rndne_f32_e32 v23, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v20, v20, v23
	v_fmac_f32_e32 v22, 0x32a5705f, v19
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v20, v22
	v_cvt_i32_f32_e32 v22, v23
	v_exp_f32_e32 v20, v20
	s_waitcnt_depctr depctr_va_vdst(0)
	v_ldexp_f32 v20, v20, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v20, 0, v20, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v19
	v_cndmask_b32_e32 v19, 0x7f800000, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v17, v21, v19
	v_dual_mov_b32 v19, v16 :: v_dual_mov_b32 v20, v17
	s_branch .LBB0_509
.LBB0_522:
	v_cmp_ne_u32_e32 vcc_lo, 1, v18
	s_cbranch_vccnz .LBB0_525
; %bb.523:                              ; %.lr.ph625
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v20
	s_mul_hi_u32 s1, s34, s15
	s_mov_b32 s4, 0
	s_and_b32 s0, vcc_lo, exec_lo
	s_cselect_b32 s0, 32, 0
	v_cndmask_b32_e64 v16, 0, 0x41b17218, vcc_lo
	v_ldexp_f32 v17, v20, s0
	s_ashr_i32 s0, s34, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_mul_i32 s2, s0, s15
	s_mul_i32 s0, s34, s15
	v_log_f32_e32 v17, v17
	s_add_i32 s1, s1, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[2:3], s[0:1], 2
	s_add_u32 s0, s8, s2
	s_addc_u32 s1, s9, s3
	s_add_u32 s2, s10, s2
	s_addc_u32 s3, s11, s3
	s_waitcnt_depctr depctr_va_vdst(0)
	v_mul_f32_e32 v18, 0x3f317217, v17
	v_cmp_gt_f32_e64 vcc_lo, 0x7f800000, |v17|
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, 0x3f317217, v17, -v18
	v_fmamk_f32 v20, v17, 0x3377d1cf, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v20
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v16, v17, v16 :: v_dual_mov_b32 v17, 0
	v_add_f32_e32 v16, v19, v16
.LBB0_524:                              ; =>This Inner Loop Header: Depth=1
	s_mov_b32 m0, s4
	s_add_i32 s4, s4, 1
	v_movrels_b32_e32 v18, v8
	v_movrels_b32_e32 v19, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v18, v18, v16
	global_store_b32 v17, v19, s[0:1]
	s_add_u32 s0, s0, 4
	s_addc_u32 s1, s1, 0
	global_store_b32 v17, v18, s[2:3]
	s_add_u32 s2, s2, 4
	s_addc_u32 s3, s3, 0
	s_cmp_lg_u32 s15, s4
	s_cbranch_scc1 .LBB0_524
.LBB0_525:
	s_endpgm
.Lfunc_end0:
	.size	mq4v2_lmhead_topk_direct_gfx1100, .Lfunc_end0-mq4v2_lmhead_topk_direct_gfx1100
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mq4v2_lmhead_topk_direct_gfx1100
		.amdhsa_group_segment_fixed_size 2304
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 328
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 72
		.amdhsa_next_free_sgpr 57
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-mq4v2_lmhead_topk_direct_gfx1100)<<4)&1008)>>4
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
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.num_vgpr, 72
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.num_agpr, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.numbered_sgpr, 57
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.num_named_barrier, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.private_seg_size, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.uses_vcc, 1
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.uses_flat_scratch, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.has_dyn_sized_stack, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.has_recursion, 0
	.set .Lmq4v2_lmhead_topk_direct_gfx1100.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 29000
; TotalNumSgprs: 59
; NumVgprs: 72
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2304 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 59
; NumVGPRsForWavesPerEU: 72
; Occupancy: 14
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
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
	.type	__hip_cuid_4b10cbb63875c105,@object ; @__hip_cuid_4b10cbb63875c105
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4b10cbb63875c105
__hip_cuid_4b10cbb63875c105:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4b10cbb63875c105, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4b10cbb63875c105
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
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         76
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         84
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         86
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         88
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         90
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         92
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         94
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         136
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 2304
    .kernarg_segment_align: 8
    .kernarg_segment_size: 328
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           mq4v2_lmhead_topk_direct_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count:     59
    .sgpr_spill_count: 0
    .symbol:         mq4v2_lmhead_topk_direct_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
