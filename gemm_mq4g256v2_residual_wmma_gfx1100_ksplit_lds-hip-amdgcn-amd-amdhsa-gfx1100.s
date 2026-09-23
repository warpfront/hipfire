	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.amdhsa_code_object_version 6
	.text
	.protected	gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds ; -- Begin function gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
	.globl	gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
	.p2align	8
	.type	gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds,@function
gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds: ; @gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v9, 15, v0
	v_lshrrev_b32_e32 v36, 5, v0
	s_lshl_b32 s1, s14, 4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v6, v8
	v_lshl_or_b32 v35, s15, 4, v9
	v_mov_b32_e32 v5, v8
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v1, v8
	s_waitcnt lgkmcnt(0)
	s_cmpk_lt_i32 s5, 0x200
	v_cmp_gt_i32_e32 vcc_lo, s6, v35
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %.lr.ph.preheader
	v_dual_cndmask_b32 v3, 0, v35 :: v_dual_mov_b32 v28, 0
	s_ashr_i32 s6, s5, 31
	v_or_b32_e32 v4, s1, v9
	s_lshr_b32 s7, s6, 24
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_i64_i32 v[1:2], null, v3, s5, 0
	s_lshr_b32 s6, s6, 23
	s_add_i32 s7, s5, s7
	s_add_i32 s5, s5, s6
	s_add_i32 s0, s4, -1
	s_ashr_i32 s5, s5, 9
	v_min_i32_e32 v3, s0, v4
	v_mul_i32_i24_e32 v37, s5, v36
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_ashr_i32 s0, s7, 8
	v_mad_i32_i24 v38, s5, v36, s5
	s_mulk_i32 s0, 0x88
	v_mul_lo_u32 v41, 0x88, v37
	v_mad_i64_i32 v[29:30], null, s0, v3, s[8:9]
	v_add_co_u32 v39, s0, s10, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v40, null, s11, v2, s0
	v_dual_mov_b32 v2, v28 :: v_dual_lshlrev_b32 v27, 8, v37
	v_mov_b32_e32 v1, v28
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v4, v28
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v6, v28
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v8, v28
	s_mov_b32 s5, 0
.LBB0_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v33, s0, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v34, null, 0, v30, s0
	v_lshlrev_b64 v[13:14], 1, v[27:28]
	v_add_nc_u32_e32 v37, 1, v37
	v_add_nc_u32_e32 v41, 0x88, v41
	s_clause 0x3
	global_load_b128 v[9:12], v[33:34], off
	global_load_b128 v[50:53], v[33:34], off offset:48
	global_load_b128 v[54:57], v[33:34], off offset:32
	global_load_b128 v[58:61], v[33:34], off offset:16
	v_add_nc_u32_e32 v27, 0x100, v27
	v_add_co_u32 v31, s0, v39, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v32, null, v40, v14, s0
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:16
	global_load_b128 v[42:45], v[31:32], off
	v_cmp_ge_i32_e64 s0, v37, v38
	s_or_b32 s5, s0, s5
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v13, 15, v11
	v_bfe_u32 v14, v11, 4, 4
	v_bfe_u32 v15, v11, 12, 4
	v_bfe_u32 v16, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_bfe_u32 v15, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v11, 24, 4
	v_lshrrev_b32_e32 v11, 28, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	v_fma_f16 v16.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v17.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 12, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v18.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 20, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v19.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fma_f16 v20.l, v9.l, v11.l, v9.h
	v_lshrrev_b32_e32 v11, 28, v12
	s_waitcnt vmcnt(2)
	v_bfe_u32 v12, v58, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v20.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v58
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[13:20], v[42:49], v[1:8]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_bfe_u32 v13, v58, 12, 4
	v_bfe_u32 v14, v58, 20, 4
	v_lshrrev_b32_e32 v15, 28, v58
	v_bfe_u32 v16, v59, 4, 4
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v58, 8, 4
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v12, v12
	v_bfe_u32 v17, v59, 12, 4
	v_bfe_u32 v18, v59, 20, 4
	v_lshrrev_b32_e32 v19, 28, v59
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v58, 16, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v58, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v59
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v59, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v59, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v59, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:48
	global_load_b128 v[19:22], v[31:32], off offset:32
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v60
	v_bfe_u32 v12, v60, 4, 4
	v_bfe_u32 v13, v60, 12, 4
	v_bfe_u32 v14, v60, 20, 4
	v_lshrrev_b32_e32 v15, 28, v60
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v60, 8, 4
	v_bfe_u32 v16, v61, 4, 4
	v_bfe_u32 v17, v61, 12, 4
	v_bfe_u32 v18, v61, 20, 4
	v_lshrrev_b32_e32 v19, 28, v61
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v60, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v60, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v61
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v61, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v61, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v61, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:80
	global_load_b128 v[19:22], v[31:32], off offset:64
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v54
	v_bfe_u32 v12, v54, 4, 4
	v_bfe_u32 v13, v54, 12, 4
	v_bfe_u32 v14, v54, 20, 4
	v_lshrrev_b32_e32 v15, 28, v54
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v54, 8, 4
	v_bfe_u32 v16, v55, 4, 4
	v_bfe_u32 v17, v55, 12, 4
	v_bfe_u32 v18, v55, 20, 4
	v_lshrrev_b32_e32 v19, 28, v55
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v54, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v54, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v55
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v55, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v55, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v55, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:112
	global_load_b128 v[19:22], v[31:32], off offset:96
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v56
	v_bfe_u32 v12, v56, 4, 4
	v_bfe_u32 v13, v56, 12, 4
	v_bfe_u32 v14, v56, 20, 4
	v_lshrrev_b32_e32 v15, 28, v56
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v56, 8, 4
	v_bfe_u32 v16, v57, 4, 4
	v_bfe_u32 v17, v57, 12, 4
	v_bfe_u32 v18, v57, 20, 4
	v_lshrrev_b32_e32 v19, 28, v57
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v56, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v56, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v57
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v57, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v57, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v57, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:144
	global_load_b128 v[19:22], v[31:32], off offset:128
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v50
	v_bfe_u32 v12, v50, 4, 4
	v_bfe_u32 v13, v50, 12, 4
	v_bfe_u32 v14, v50, 20, 4
	v_lshrrev_b32_e32 v15, 28, v50
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v50, 8, 4
	v_bfe_u32 v16, v51, 4, 4
	v_bfe_u32 v17, v51, 12, 4
	v_bfe_u32 v18, v51, 20, 4
	v_lshrrev_b32_e32 v19, 28, v51
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v50, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v50, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v51
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v51, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v51, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v51, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:176
	global_load_b128 v[19:22], v[31:32], off offset:160
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v52
	v_bfe_u32 v12, v52, 4, 4
	v_bfe_u32 v13, v52, 12, 4
	v_bfe_u32 v14, v52, 20, 4
	v_lshrrev_b32_e32 v15, 28, v52
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v52, 8, 4
	v_bfe_u32 v16, v53, 4, 4
	v_bfe_u32 v17, v53, 12, 4
	v_bfe_u32 v18, v53, 20, 4
	v_lshrrev_b32_e32 v19, 28, v53
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v52, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v52, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v53
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v53, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v53, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v53, 24, 4
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:240
	global_load_b128 v[50:53], v[31:32], off offset:224
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:208
	global_load_b128 v[19:22], v[31:32], off offset:192
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_clause 0x3
	global_load_b128 v[11:14], v[33:34], off offset:112
	global_load_b128 v[15:18], v[33:34], off offset:96
	global_load_b128 v[19:22], v[33:34], off offset:80
	global_load_b128 v[23:26], v[33:34], off offset:64
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v42, 15, v23
	v_bfe_u32 v43, v23, 4, 4
	v_bfe_u32 v44, v23, 12, 4
	v_bfe_u32 v45, v23, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v42.l, v42
	v_cvt_f16_f32_e32 v42.h, v43
	v_bfe_u32 v43, v23, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v42.l, v9.l, v42.l, v9.h
	v_fma_f16 v42.h, v9.l, v42.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v43, v43
	v_cvt_f16_f32_e32 v43.l, v43
	v_cvt_f16_f32_e32 v43.h, v44
	v_bfe_u32 v44, v23, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v43.l, v9.l, v43.l, v9.h
	v_fma_f16 v43.h, v9.l, v43.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f16_f32_e32 v44.l, v44
	v_cvt_f16_f32_e32 v44.h, v45
	v_bfe_u32 v45, v23, 24, 4
	v_lshrrev_b32_e32 v23, 28, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v44.l, v9.l, v44.l, v9.h
	v_fma_f16 v44.h, v9.l, v44.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v45.l, v45
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v45.l, v9.l, v45.l, v9.h
	v_fma_f16 v45.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v23, 15, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 4, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v46.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 12, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v47.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 20, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v48.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v9.l, v23.l, v9.h
	v_lshrrev_b32_e32 v23, 28, v24
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v49.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v9, 15, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:272
	global_load_b128 v[50:53], v[31:32], off offset:256
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v25
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v26
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:304
	global_load_b128 v[50:53], v[31:32], off offset:288
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v20
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v21
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v22
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v22
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:336
	global_load_b128 v[19:22], v[31:32], off offset:320
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[19:26], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:368
	global_load_b128 v[42:45], v[31:32], off offset:352
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v16
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v17
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:400
	global_load_b128 v[42:45], v[31:32], off offset:384
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v11
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:432
	global_load_b128 v[42:45], v[31:32], off offset:416
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v12
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v13
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:464
	global_load_b128 v[42:45], v[31:32], off offset:448
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v14
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	global_load_b64 v[18:19], v[33:34], off offset:128
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v9, 15, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v11.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v11.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v12.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v12.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v13.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v13.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v14.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v14.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:496
	global_load_b128 v[19:22], v[31:32], off offset:480
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fmac_f16_e32 v10.h, v10.l, v9.l
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mov_b16_e32 v18.h, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_and_not1_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB0_2
; %bb.3:                                ; %Flow255
	s_or_b32 exec_lo, exec_lo, s5
.LBB0_4:                                ; %Flow256
	v_and_b32_e32 v9, 31, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 2, v9
	v_lshl_or_b32 v9, v36, 10, v18
	ds_store_2addr_b32 v9, v1, v2 offset1:32
	ds_store_2addr_b32 v9, v3, v4 offset0:64 offset1:96
	ds_store_2addr_b32 v9, v5, v6 offset0:128 offset1:160
	ds_store_2addr_b32 v9, v7, v8 offset0:192 offset1:224
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v36
	s_cbranch_execz .LBB0_22
; %bb.5:                                ; %.preheader177
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_22
; %bb.6:                                ; %.preheader
	v_add_nc_u32_e32 v3, 0x400, v18
	v_add_nc_u32_e32 v1, 0x80, v18
	v_mad_i64_i32 v[5:6], null, s4, v35, 0
	ds_load_2addr_b32 v[11:12], v18 offset0:96 offset1:128
	ds_load_2addr_b32 v[7:8], v18 offset0:160 offset1:192
	ds_load_2addr_stride64_b32 v[1:2], v1 offset0:3 offset1:4
	ds_load_2addr_b32 v[13:14], v3 offset0:64 offset1:96
	ds_load_2addr_b32 v[9:10], v3 offset0:128 offset1:160
	ds_load_2addr_b32 v[3:4], v3 offset0:192 offset1:224
	ds_load_2addr_b32 v[15:16], v18 offset0:32 offset1:64
	v_lshrrev_b32_e32 v0, 4, v0
	s_mov_b32 s0, exec_lo
	v_lshlrev_b64 v[19:20], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v5, s1, v0
	v_add_co_u32 v0, vcc_lo, s2, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v17, null, s3, v20, vcc_lo
	v_ashrrev_i32_e32 v6, 31, v5
	v_cmpx_gt_i32_e64 s4, v5
	s_cbranch_execz .LBB0_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[19:20], 2, v[5:6]
	ds_load_2addr_stride64_b32 v[21:22], v18 offset1:4
	v_add_co_u32 v19, vcc_lo, v0, v19
	v_add_co_ci_u32_e64 v20, null, v17, v20, vcc_lo
	global_load_b32 v23, v[19:20], off
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v18, 0, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v18, v22
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v18, v18, v23
	global_store_b32 v[19:20], v18, off
.LBB0_8:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v18, 2, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v18
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_lshlrev_b64 v[18:19], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v18, vcc_lo, v0, v18
	v_add_co_ci_u32_e64 v19, null, v17, v19, vcc_lo
	global_load_b32 v20, v[18:19], off offset:8
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v15, 0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v15, v2
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v2, v20
	global_store_b32 v[18:19], v2, off offset:8
.LBB0_10:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(4)
	v_or_b32_e32 v2, 4, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v2
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_lshlrev_b64 v[18:19], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v18, vcc_lo, v0, v18
	v_add_co_ci_u32_e64 v19, null, v17, v19, vcc_lo
	global_load_b32 v2, v[18:19], off offset:16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v15, 0, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v13, v15, v13
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v13, v2
	global_store_b32 v[18:19], v2, off offset:16
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 6, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v2
	s_cbranch_execz .LBB0_14
; %bb.13:
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b64 v[15:16], 2, v[5:6]
	v_add_f32_e32 v11, 0, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v11, v14
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v16, null, v17, v16, vcc_lo
	global_load_b32 v2, v[15:16], off offset:24
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v11, v2
	global_store_b32 v[15:16], v2, off offset:24
.LBB0_14:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 8, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v2
	s_cbranch_execz .LBB0_16
; %bb.15:
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b64 v[13:14], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v13, vcc_lo, v0, v13
	v_add_co_ci_u32_e64 v14, null, v17, v14, vcc_lo
	global_load_b32 v2, v[13:14], off offset:32
	v_add_f32_e32 v11, 0, v12
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v11, v9
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v9, v2
	global_store_b32 v[13:14], v2, off offset:32
.LBB0_16:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 10, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v2
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_lshlrev_b64 v[11:12], 2, v[5:6]
	v_add_f32_e32 v7, 0, v7
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v7, v10
	v_add_co_u32 v11, vcc_lo, v0, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v12, null, v17, v12, vcc_lo
	global_load_b32 v2, v[11:12], off offset:40
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v7, v2
	global_store_b32 v[11:12], v2, off offset:40
.LBB0_18:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 12, v5
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v2
	s_cbranch_execz .LBB0_20
; %bb.19:
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b64 v[9:10], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v9, vcc_lo, v0, v9
	v_add_co_ci_u32_e64 v10, null, v17, v10, vcc_lo
	global_load_b32 v2, v[9:10], off offset:48
	v_add_f32_e32 v7, 0, v8
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v3, v7, v3
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v3, v2
	global_store_b32 v[9:10], v2, off offset:48
.LBB0_20:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v2, 14, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s4, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_22
; %bb.21:
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b64 v[2:3], 2, v[5:6]
	v_add_f32_e32 v1, 0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v1, v4
	v_add_co_u32 v2, vcc_lo, v0, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, v17, v3, vcc_lo
	global_load_b32 v0, v[2:3], off offset:56
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v0, v1, v0
	global_store_b32 v[2:3], v0, off offset:56
.LBB0_22:                               ; %.loopexit
	s_endpgm
.Lfunc_end0:
	.size	gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds, .Lfunc_end0-gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
		.amdhsa_group_segment_fixed_size 2048
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 14
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
		.amdhsa_next_free_vgpr 62
		.amdhsa_next_free_sgpr 16
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds)<<4)&1008)>>4
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
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.num_vgpr, 62
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.numbered_sgpr, 16
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9080
; TotalNumSgprs: 18
; NumVgprs: 62
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2048 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 62
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds ; -- Begin function gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
	.globl	gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
	.p2align	8
	.type	gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds,@function
gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds: ; @gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v9, 15, v0
	v_lshrrev_b32_e32 v36, 5, v0
	s_lshl_b32 s1, s14, 4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v6, v8
	v_lshl_or_b32 v35, s15, 4, v9
	v_mov_b32_e32 v5, v8
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v1, v8
	s_waitcnt lgkmcnt(0)
	s_cmpk_lt_i32 s5, 0x400
	v_cmp_gt_i32_e32 vcc_lo, s6, v35
	s_cbranch_scc1 .LBB1_4
; %bb.1:                                ; %.lr.ph.preheader
	v_dual_cndmask_b32 v3, 0, v35 :: v_dual_mov_b32 v28, 0
	s_ashr_i32 s6, s5, 31
	v_or_b32_e32 v4, s1, v9
	s_lshr_b32 s7, s6, 24
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_i64_i32 v[1:2], null, v3, s5, 0
	s_lshr_b32 s6, s6, 22
	s_add_i32 s7, s5, s7
	s_add_i32 s5, s5, s6
	s_add_i32 s0, s4, -1
	s_ashr_i32 s5, s5, 10
	v_min_i32_e32 v3, s0, v4
	v_mul_i32_i24_e32 v37, s5, v36
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_ashr_i32 s0, s7, 8
	v_mad_i32_i24 v38, s5, v36, s5
	s_mulk_i32 s0, 0x88
	v_mul_lo_u32 v41, 0x88, v37
	v_mad_i64_i32 v[29:30], null, s0, v3, s[8:9]
	v_add_co_u32 v39, s0, s10, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v40, null, s11, v2, s0
	v_dual_mov_b32 v2, v28 :: v_dual_lshlrev_b32 v27, 8, v37
	v_mov_b32_e32 v1, v28
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v4, v28
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v6, v28
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v8, v28
	s_mov_b32 s5, 0
.LBB1_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v33, s0, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v34, null, 0, v30, s0
	v_lshlrev_b64 v[13:14], 1, v[27:28]
	v_add_nc_u32_e32 v37, 1, v37
	v_add_nc_u32_e32 v41, 0x88, v41
	s_clause 0x3
	global_load_b128 v[9:12], v[33:34], off
	global_load_b128 v[50:53], v[33:34], off offset:48
	global_load_b128 v[54:57], v[33:34], off offset:32
	global_load_b128 v[58:61], v[33:34], off offset:16
	v_add_nc_u32_e32 v27, 0x100, v27
	v_add_co_u32 v31, s0, v39, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v32, null, v40, v14, s0
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:16
	global_load_b128 v[42:45], v[31:32], off
	v_cmp_ge_i32_e64 s0, v37, v38
	s_or_b32 s5, s0, s5
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v13, 15, v11
	v_bfe_u32 v14, v11, 4, 4
	v_bfe_u32 v15, v11, 12, 4
	v_bfe_u32 v16, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_bfe_u32 v15, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v11, 24, 4
	v_lshrrev_b32_e32 v11, 28, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	v_fma_f16 v16.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v17.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 12, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v18.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 20, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v19.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fma_f16 v20.l, v9.l, v11.l, v9.h
	v_lshrrev_b32_e32 v11, 28, v12
	s_waitcnt vmcnt(2)
	v_bfe_u32 v12, v58, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v20.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v58
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[13:20], v[42:49], v[1:8]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_bfe_u32 v13, v58, 12, 4
	v_bfe_u32 v14, v58, 20, 4
	v_lshrrev_b32_e32 v15, 28, v58
	v_bfe_u32 v16, v59, 4, 4
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v58, 8, 4
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v12, v12
	v_bfe_u32 v17, v59, 12, 4
	v_bfe_u32 v18, v59, 20, 4
	v_lshrrev_b32_e32 v19, 28, v59
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v58, 16, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v58, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v59
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v59, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v59, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v59, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:48
	global_load_b128 v[19:22], v[31:32], off offset:32
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v60
	v_bfe_u32 v12, v60, 4, 4
	v_bfe_u32 v13, v60, 12, 4
	v_bfe_u32 v14, v60, 20, 4
	v_lshrrev_b32_e32 v15, 28, v60
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v60, 8, 4
	v_bfe_u32 v16, v61, 4, 4
	v_bfe_u32 v17, v61, 12, 4
	v_bfe_u32 v18, v61, 20, 4
	v_lshrrev_b32_e32 v19, 28, v61
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v60, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v60, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v61
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v61, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v61, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v61, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:80
	global_load_b128 v[19:22], v[31:32], off offset:64
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v54
	v_bfe_u32 v12, v54, 4, 4
	v_bfe_u32 v13, v54, 12, 4
	v_bfe_u32 v14, v54, 20, 4
	v_lshrrev_b32_e32 v15, 28, v54
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v54, 8, 4
	v_bfe_u32 v16, v55, 4, 4
	v_bfe_u32 v17, v55, 12, 4
	v_bfe_u32 v18, v55, 20, 4
	v_lshrrev_b32_e32 v19, 28, v55
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v54, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v54, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v55
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v55, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v55, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v55, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:112
	global_load_b128 v[19:22], v[31:32], off offset:96
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v56
	v_bfe_u32 v12, v56, 4, 4
	v_bfe_u32 v13, v56, 12, 4
	v_bfe_u32 v14, v56, 20, 4
	v_lshrrev_b32_e32 v15, 28, v56
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v56, 8, 4
	v_bfe_u32 v16, v57, 4, 4
	v_bfe_u32 v17, v57, 12, 4
	v_bfe_u32 v18, v57, 20, 4
	v_lshrrev_b32_e32 v19, 28, v57
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v56, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v56, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v57
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v57, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v57, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v57, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:144
	global_load_b128 v[19:22], v[31:32], off offset:128
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v50
	v_bfe_u32 v12, v50, 4, 4
	v_bfe_u32 v13, v50, 12, 4
	v_bfe_u32 v14, v50, 20, 4
	v_lshrrev_b32_e32 v15, 28, v50
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v50, 8, 4
	v_bfe_u32 v16, v51, 4, 4
	v_bfe_u32 v17, v51, 12, 4
	v_bfe_u32 v18, v51, 20, 4
	v_lshrrev_b32_e32 v19, 28, v51
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v50, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v50, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v51
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v51, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v51, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v51, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:176
	global_load_b128 v[19:22], v[31:32], off offset:160
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v52
	v_bfe_u32 v12, v52, 4, 4
	v_bfe_u32 v13, v52, 12, 4
	v_bfe_u32 v14, v52, 20, 4
	v_lshrrev_b32_e32 v15, 28, v52
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v52, 8, 4
	v_bfe_u32 v16, v53, 4, 4
	v_bfe_u32 v17, v53, 12, 4
	v_bfe_u32 v18, v53, 20, 4
	v_lshrrev_b32_e32 v19, 28, v53
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v52, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v52, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v53
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v53, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v53, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v53, 24, 4
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:240
	global_load_b128 v[50:53], v[31:32], off offset:224
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:208
	global_load_b128 v[19:22], v[31:32], off offset:192
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_clause 0x3
	global_load_b128 v[11:14], v[33:34], off offset:112
	global_load_b128 v[15:18], v[33:34], off offset:96
	global_load_b128 v[19:22], v[33:34], off offset:80
	global_load_b128 v[23:26], v[33:34], off offset:64
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v42, 15, v23
	v_bfe_u32 v43, v23, 4, 4
	v_bfe_u32 v44, v23, 12, 4
	v_bfe_u32 v45, v23, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v42.l, v42
	v_cvt_f16_f32_e32 v42.h, v43
	v_bfe_u32 v43, v23, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v42.l, v9.l, v42.l, v9.h
	v_fma_f16 v42.h, v9.l, v42.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v43, v43
	v_cvt_f16_f32_e32 v43.l, v43
	v_cvt_f16_f32_e32 v43.h, v44
	v_bfe_u32 v44, v23, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v43.l, v9.l, v43.l, v9.h
	v_fma_f16 v43.h, v9.l, v43.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f16_f32_e32 v44.l, v44
	v_cvt_f16_f32_e32 v44.h, v45
	v_bfe_u32 v45, v23, 24, 4
	v_lshrrev_b32_e32 v23, 28, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v44.l, v9.l, v44.l, v9.h
	v_fma_f16 v44.h, v9.l, v44.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v45.l, v45
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v45.l, v9.l, v45.l, v9.h
	v_fma_f16 v45.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v23, 15, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 4, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v46.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 12, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v47.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 20, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v48.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v9.l, v23.l, v9.h
	v_lshrrev_b32_e32 v23, 28, v24
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v49.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v9, 15, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:272
	global_load_b128 v[50:53], v[31:32], off offset:256
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v25
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v26
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:304
	global_load_b128 v[50:53], v[31:32], off offset:288
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v20
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v21
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v22
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v22
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:336
	global_load_b128 v[19:22], v[31:32], off offset:320
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[19:26], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:368
	global_load_b128 v[42:45], v[31:32], off offset:352
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v16
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v17
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:400
	global_load_b128 v[42:45], v[31:32], off offset:384
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v11
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:432
	global_load_b128 v[42:45], v[31:32], off offset:416
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v12
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v13
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:464
	global_load_b128 v[42:45], v[31:32], off offset:448
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v14
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	global_load_b64 v[18:19], v[33:34], off offset:128
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v9, 15, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v11.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v11.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v12.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v12.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v13.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v13.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v14.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v14.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:496
	global_load_b128 v[19:22], v[31:32], off offset:480
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fmac_f16_e32 v10.h, v10.l, v9.l
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mov_b16_e32 v18.h, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_and_not1_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB1_2
; %bb.3:                                ; %Flow255
	s_or_b32 exec_lo, exec_lo, s5
.LBB1_4:                                ; %Flow256
	v_and_b32_e32 v9, 31, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v32, 2, v9
	v_lshl_or_b32 v9, v36, 10, v32
	ds_store_2addr_b32 v9, v1, v2 offset1:32
	ds_store_2addr_b32 v9, v3, v4 offset0:64 offset1:96
	ds_store_2addr_b32 v9, v5, v6 offset0:128 offset1:160
	ds_store_2addr_b32 v9, v7, v8 offset0:192 offset1:224
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v36
	s_cbranch_execz .LBB1_22
; %bb.5:                                ; %.preheader177
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_22
; %bb.6:                                ; %.preheader
	v_add_nc_u32_e32 v1, 0x400, v32
	v_add_nc_u32_e32 v3, 0x800, v32
	v_add_nc_u32_e32 v7, 0xc00, v32
	v_add_nc_u32_e32 v4, 0x80, v32
	v_mad_i64_i32 v[11:12], null, s4, v35, 0
	ds_load_2addr_b32 v[23:24], v1 offset0:64 offset1:96
	ds_load_2addr_b32 v[15:16], v1 offset0:128 offset1:160
	ds_load_2addr_b32 v[1:2], v1 offset0:192 offset1:224
	ds_load_2addr_b32 v[29:30], v3 offset0:32 offset1:64
	ds_load_2addr_b32 v[21:22], v3 offset0:96 offset1:128
	ds_load_2addr_b32 v[13:14], v3 offset0:160 offset1:192
	ds_load_2addr_stride64_b32 v[5:6], v4 offset0:3 offset1:4
	ds_load_2addr_stride64_b32 v[3:4], v4 offset0:11 offset1:12
	ds_load_2addr_b32 v[25:26], v7 offset0:64 offset1:96
	ds_load_2addr_b32 v[17:18], v7 offset0:128 offset1:160
	ds_load_2addr_b32 v[7:8], v7 offset0:192 offset1:224
	ds_load_2addr_b32 v[27:28], v32 offset0:32 offset1:64
	ds_load_2addr_b32 v[19:20], v32 offset0:96 offset1:128
	ds_load_2addr_b32 v[9:10], v32 offset0:160 offset1:192
	v_lshrrev_b32_e32 v0, 4, v0
	v_lshlrev_b64 v[33:34], 2, v[11:12]
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v11, s1, v0
	v_add_co_u32 v0, vcc_lo, s2, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v31, null, s3, v34, vcc_lo
	v_ashrrev_i32_e32 v12, 31, v11
	v_cmpx_gt_i32_e64 s4, v11
	s_cbranch_execz .LBB1_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[33:34], 2, v[11:12]
	ds_load_2addr_stride64_b32 v[35:36], v32 offset1:4
	ds_load_2addr_stride64_b32 v[37:38], v32 offset0:8 offset1:12
	v_add_co_u32 v33, vcc_lo, v0, v33
	v_add_co_ci_u32_e64 v34, null, v31, v34, vcc_lo
	global_load_b32 v39, v[33:34], off
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v32, 0, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v36
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v32, v32, v37
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v32, v32, v38
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v32, v32, v39
	global_store_b32 v[33:34], v32, off
.LBB1_8:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v32, 2, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v32
	s_cbranch_execz .LBB1_10
; %bb.9:
	v_lshlrev_b64 v[32:33], 2, v[11:12]
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v27, 0, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v27, v6
	v_add_co_u32 v32, vcc_lo, v0, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v33, null, v31, v33, vcc_lo
	v_add_f32_e32 v6, v6, v29
	global_load_b32 v34, v[32:33], off offset:8
	v_add_f32_e32 v4, v6, v4
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v34
	global_store_b32 v[32:33], v4, off offset:8
.LBB1_10:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(6)
	v_or_b32_e32 v4, 4, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v4
	s_cbranch_execz .LBB1_12
; %bb.11:
	v_lshlrev_b64 v[32:33], 2, v[11:12]
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v6, 0, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v6, v23
	v_add_co_u32 v32, vcc_lo, v0, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v33, null, v31, v33, vcc_lo
	v_add_f32_e32 v6, v6, v30
	global_load_b32 v4, v[32:33], off offset:16
	v_add_f32_e32 v6, v6, v25
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v4, v6, v4
	global_store_b32 v[32:33], v4, off offset:16
.LBB1_12:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v4, 6, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v4
	s_cbranch_execz .LBB1_14
; %bb.13:
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b64 v[27:28], 2, v[11:12]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, 0, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v6, v24
	v_add_co_u32 v27, vcc_lo, v0, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v28, null, v31, v28, vcc_lo
	v_add_f32_e32 v6, v6, v21
	global_load_b32 v4, v[27:28], off offset:24
	v_add_f32_e32 v6, v6, v26
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v4, v6, v4
	global_store_b32 v[27:28], v4, off offset:24
.LBB1_14:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v4, 8, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v4
	s_cbranch_execz .LBB1_16
; %bb.15:
	v_lshlrev_b64 v[23:24], 2, v[11:12]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, 0, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v6, v15
	v_add_co_u32 v23, vcc_lo, v0, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v24, null, v31, v24, vcc_lo
	v_add_f32_e32 v6, v6, v22
	global_load_b32 v4, v[23:24], off offset:32
	v_add_f32_e32 v6, v6, v17
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v4, v6, v4
	global_store_b32 v[23:24], v4, off offset:32
.LBB1_16:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v4, 10, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v4
	s_cbranch_execz .LBB1_18
; %bb.17:
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b64 v[19:20], 2, v[11:12]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, 0, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v6, v16
	v_add_co_u32 v19, vcc_lo, v0, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v20, null, v31, v20, vcc_lo
	v_add_f32_e32 v6, v6, v13
	global_load_b32 v4, v[19:20], off offset:40
	v_add_f32_e32 v6, v6, v18
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v4, v6, v4
	global_store_b32 v[19:20], v4, off offset:40
.LBB1_18:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v4, 12, v11
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v4
	s_cbranch_execz .LBB1_20
; %bb.19:
	v_lshlrev_b64 v[15:16], 2, v[11:12]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, 0, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v6, v1
	v_add_co_u32 v15, vcc_lo, v0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v16, null, v31, v16, vcc_lo
	v_add_f32_e32 v1, v1, v14
	global_load_b32 v4, v[15:16], off offset:48
	v_add_f32_e32 v1, v1, v7
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v4
	global_store_b32 v[15:16], v1, off offset:48
.LBB1_20:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v1, 14, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s4, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB1_22
; %bb.21:
	s_waitcnt lgkmcnt(3)
	v_lshlrev_b64 v[6:7], 2, v[11:12]
	v_add_f32_e32 v5, 0, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v5, v2
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, v31, v7, vcc_lo
	v_add_f32_e32 v2, v2, v3
	global_load_b32 v4, v[0:1], off offset:56
	v_add_f32_e32 v2, v2, v8
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v4
	global_store_b32 v[0:1], v2, off offset:56
.LBB1_22:                               ; %.loopexit
	s_endpgm
.Lfunc_end1:
	.size	gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds, .Lfunc_end1-gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 14
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
		.amdhsa_next_free_vgpr 62
		.amdhsa_next_free_sgpr 16
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds)<<4)&1008)>>4
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
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.num_vgpr, 62
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.numbered_sgpr, 16
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9260
; TotalNumSgprs: 18
; NumVgprs: 62
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 62
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds ; -- Begin function gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
	.globl	gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
	.p2align	8
	.type	gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds,@function
gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds: ; @gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b128 s[4:7], s[0:1], 0x18
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v9, 15, v0
	v_lshrrev_b32_e32 v36, 5, v0
	s_lshl_b32 s1, s14, 4
	s_delay_alu instid0(VALU_DEP_2)
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v6, v8
	v_lshl_or_b32 v35, s15, 4, v9
	v_mov_b32_e32 v5, v8
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v1, v8
	s_waitcnt lgkmcnt(0)
	s_cmpk_lt_i32 s5, 0x800
	v_cmp_gt_i32_e32 vcc_lo, s6, v35
	s_cbranch_scc1 .LBB2_4
; %bb.1:                                ; %.lr.ph.preheader
	v_dual_cndmask_b32 v3, 0, v35 :: v_dual_mov_b32 v28, 0
	s_ashr_i32 s6, s5, 31
	v_or_b32_e32 v4, s1, v9
	s_lshr_b32 s7, s6, 24
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_i64_i32 v[1:2], null, v3, s5, 0
	s_lshr_b32 s6, s6, 21
	s_add_i32 s7, s5, s7
	s_add_i32 s5, s5, s6
	s_add_i32 s0, s4, -1
	s_ashr_i32 s5, s5, 11
	v_min_i32_e32 v3, s0, v4
	v_mul_i32_i24_e32 v37, s5, v36
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_ashr_i32 s0, s7, 8
	v_mad_i32_i24 v38, s5, v36, s5
	s_mulk_i32 s0, 0x88
	v_mul_lo_u32 v41, 0x88, v37
	v_mad_i64_i32 v[29:30], null, s0, v3, s[8:9]
	v_add_co_u32 v39, s0, s10, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v40, null, s11, v2, s0
	v_dual_mov_b32 v2, v28 :: v_dual_lshlrev_b32 v27, 8, v37
	v_mov_b32_e32 v1, v28
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v4, v28
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v6, v28
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v8, v28
	s_mov_b32 s5, 0
.LBB2_2:                                ; %.lr.ph
                                        ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v33, s0, v29, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v34, null, 0, v30, s0
	v_lshlrev_b64 v[13:14], 1, v[27:28]
	v_add_nc_u32_e32 v37, 1, v37
	v_add_nc_u32_e32 v41, 0x88, v41
	s_clause 0x3
	global_load_b128 v[9:12], v[33:34], off
	global_load_b128 v[50:53], v[33:34], off offset:48
	global_load_b128 v[54:57], v[33:34], off offset:32
	global_load_b128 v[58:61], v[33:34], off offset:16
	v_add_nc_u32_e32 v27, 0x100, v27
	v_add_co_u32 v31, s0, v39, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v32, null, v40, v14, s0
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:16
	global_load_b128 v[42:45], v[31:32], off
	v_cmp_ge_i32_e64 s0, v37, v38
	s_or_b32 s5, s0, s5
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v13, 15, v11
	v_bfe_u32 v14, v11, 4, 4
	v_bfe_u32 v15, v11, 12, 4
	v_bfe_u32 v16, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_bfe_u32 v15, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v11, 24, 4
	v_lshrrev_b32_e32 v11, 28, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	v_fma_f16 v16.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v17.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 12, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v18.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.l, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 20, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11.l, v11
	v_fma_f16 v19.h, v9.l, v11.l, v9.h
	v_bfe_u32 v11, v12, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fma_f16 v20.l, v9.l, v11.l, v9.h
	v_lshrrev_b32_e32 v11, 28, v12
	s_waitcnt vmcnt(2)
	v_bfe_u32 v12, v58, 4, 4
	v_cvt_f32_ubyte0_e32 v11, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f16_f32_e32 v11.l, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_fma_f16 v20.h, v9.l, v11.l, v9.h
	v_and_b32_e32 v11, 15, v58
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[13:20], v[42:49], v[1:8]
	s_delay_alu instid0(VALU_DEP_2)
	v_cvt_f32_ubyte0_e32 v11, v11
	v_bfe_u32 v13, v58, 12, 4
	v_bfe_u32 v14, v58, 20, 4
	v_lshrrev_b32_e32 v15, 28, v58
	v_bfe_u32 v16, v59, 4, 4
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v58, 8, 4
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v12, v12
	v_bfe_u32 v17, v59, 12, 4
	v_bfe_u32 v18, v59, 20, 4
	v_lshrrev_b32_e32 v19, 28, v59
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v58, 16, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v58, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v59
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v59, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v59, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v59, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:48
	global_load_b128 v[19:22], v[31:32], off offset:32
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v60
	v_bfe_u32 v12, v60, 4, 4
	v_bfe_u32 v13, v60, 12, 4
	v_bfe_u32 v14, v60, 20, 4
	v_lshrrev_b32_e32 v15, 28, v60
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v60, 8, 4
	v_bfe_u32 v16, v61, 4, 4
	v_bfe_u32 v17, v61, 12, 4
	v_bfe_u32 v18, v61, 20, 4
	v_lshrrev_b32_e32 v19, 28, v61
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v60, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v60, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v61
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v61, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v61, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v61, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:80
	global_load_b128 v[19:22], v[31:32], off offset:64
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v54
	v_bfe_u32 v12, v54, 4, 4
	v_bfe_u32 v13, v54, 12, 4
	v_bfe_u32 v14, v54, 20, 4
	v_lshrrev_b32_e32 v15, 28, v54
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v54, 8, 4
	v_bfe_u32 v16, v55, 4, 4
	v_bfe_u32 v17, v55, 12, 4
	v_bfe_u32 v18, v55, 20, 4
	v_lshrrev_b32_e32 v19, 28, v55
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v54, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v54, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v55
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v55, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v55, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v55, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:112
	global_load_b128 v[19:22], v[31:32], off offset:96
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v56
	v_bfe_u32 v12, v56, 4, 4
	v_bfe_u32 v13, v56, 12, 4
	v_bfe_u32 v14, v56, 20, 4
	v_lshrrev_b32_e32 v15, 28, v56
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v56, 8, 4
	v_bfe_u32 v16, v57, 4, 4
	v_bfe_u32 v17, v57, 12, 4
	v_bfe_u32 v18, v57, 20, 4
	v_lshrrev_b32_e32 v19, 28, v57
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v56, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v56, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v57
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v57, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v57, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v57, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:144
	global_load_b128 v[19:22], v[31:32], off offset:128
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v50
	v_bfe_u32 v12, v50, 4, 4
	v_bfe_u32 v13, v50, 12, 4
	v_bfe_u32 v14, v50, 20, 4
	v_lshrrev_b32_e32 v15, 28, v50
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v50, 8, 4
	v_bfe_u32 v16, v51, 4, 4
	v_bfe_u32 v17, v51, 12, 4
	v_bfe_u32 v18, v51, 20, 4
	v_lshrrev_b32_e32 v19, 28, v51
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v50, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v50, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v51
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v51, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v51, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v51, 24, 4
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:176
	global_load_b128 v[19:22], v[31:32], off offset:160
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	v_and_b32_e32 v11, 15, v52
	v_bfe_u32 v12, v52, 4, 4
	v_bfe_u32 v13, v52, 12, 4
	v_bfe_u32 v14, v52, 20, 4
	v_lshrrev_b32_e32 v15, 28, v52
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f16_f32_e32 v11.l, v11
	v_cvt_f16_f32_e32 v11.h, v12
	v_bfe_u32 v12, v52, 8, 4
	v_bfe_u32 v16, v53, 4, 4
	v_bfe_u32 v17, v53, 12, 4
	v_bfe_u32 v18, v53, 20, 4
	v_lshrrev_b32_e32 v19, 28, v53
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f16_f32_e32 v12.l, v12
	v_cvt_f16_f32_e32 v12.h, v13
	v_bfe_u32 v13, v52, 16, 4
	v_fma_f16 v11.l, v9.l, v11.l, v9.h
	v_fma_f16 v11.h, v9.l, v11.h, v9.h
	v_fma_f16 v12.l, v9.l, v12.l, v9.h
	v_fma_f16 v12.h, v9.l, v12.h, v9.h
	v_cvt_f32_ubyte0_e32 v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v13.l, v13
	v_cvt_f16_f32_e32 v13.h, v14
	v_bfe_u32 v14, v52, 24, 4
	v_fma_f16 v13.l, v9.l, v13.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v13.h, v9.l, v13.h, v9.h
	v_cvt_f32_ubyte0_e32 v14, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v14.l, v14
	v_cvt_f16_f32_e32 v14.h, v15
	v_and_b32_e32 v15, 15, v53
	v_fma_f16 v14.l, v9.l, v14.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v14.h, v9.l, v14.h, v9.h
	v_cvt_f32_ubyte0_e32 v15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v15.l, v15
	v_cvt_f16_f32_e32 v15.h, v16
	v_bfe_u32 v16, v53, 8, 4
	v_fma_f16 v15.l, v9.l, v15.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v15.h, v9.l, v15.h, v9.h
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_cvt_f16_f32_e32 v16.l, v16
	v_cvt_f16_f32_e32 v16.h, v17
	v_bfe_u32 v17, v53, 16, 4
	v_fma_f16 v16.l, v9.l, v16.l, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v16.h, v9.l, v16.h, v9.h
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v17.l, v17
	v_cvt_f16_f32_e32 v17.h, v18
	v_bfe_u32 v18, v53, 24, 4
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:240
	global_load_b128 v[50:53], v[31:32], off offset:224
	v_fma_f16 v17.l, v9.l, v17.l, v9.h
	v_fma_f16 v17.h, v9.l, v17.h, v9.h
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18.l, v18
	v_cvt_f16_f32_e32 v18.h, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:208
	global_load_b128 v[19:22], v[31:32], off offset:192
	v_fma_f16 v18.l, v9.l, v18.l, v9.h
	v_fma_f16 v18.h, v9.l, v18.h, v9.h
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_clause 0x3
	global_load_b128 v[11:14], v[33:34], off offset:112
	global_load_b128 v[15:18], v[33:34], off offset:96
	global_load_b128 v[19:22], v[33:34], off offset:80
	global_load_b128 v[23:26], v[33:34], off offset:64
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v42, 15, v23
	v_bfe_u32 v43, v23, 4, 4
	v_bfe_u32 v44, v23, 12, 4
	v_bfe_u32 v45, v23, 20, 4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v42, v42
	v_cvt_f32_ubyte0_e32 v43, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f32_ubyte0_e32 v45, v45
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f16_f32_e32 v42.l, v42
	v_cvt_f16_f32_e32 v42.h, v43
	v_bfe_u32 v43, v23, 8, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v42.l, v9.l, v42.l, v9.h
	v_fma_f16 v42.h, v9.l, v42.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v43, v43
	v_cvt_f16_f32_e32 v43.l, v43
	v_cvt_f16_f32_e32 v43.h, v44
	v_bfe_u32 v44, v23, 16, 4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f16 v43.l, v9.l, v43.l, v9.h
	v_fma_f16 v43.h, v9.l, v43.h, v9.h
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v44, v44
	v_cvt_f16_f32_e32 v44.l, v44
	v_cvt_f16_f32_e32 v44.h, v45
	v_bfe_u32 v45, v23, 24, 4
	v_lshrrev_b32_e32 v23, 28, v23
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f16 v44.l, v9.l, v44.l, v9.h
	v_fma_f16 v44.h, v9.l, v44.h, v9.h
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v45
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v45.l, v45
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f16 v45.l, v9.l, v45.l, v9.h
	v_fma_f16 v45.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v23, 15, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 4, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v46.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 12, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v47.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 20, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v48.h, v9.l, v23.l, v9.h
	v_bfe_u32 v23, v24, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f16_f32_e32 v23.l, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v9.l, v23.l, v9.h
	v_lshrrev_b32_e32 v23, 28, v24
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23.l, v23
	v_fma_f16 v49.h, v9.l, v23.l, v9.h
	v_and_b32_e32 v9, 15, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:272
	global_load_b128 v[50:53], v[31:32], off offset:256
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v25, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v25
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v26, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v26
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	s_clause 0x1
	global_load_b128 v[54:57], v[31:32], off offset:304
	global_load_b128 v[50:53], v[31:32], off offset:288
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v20
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v20, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v21
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[50:57], v[1:8]
	v_fma_f16 v42.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v42.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v43.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v43.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v44.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v44.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v21, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v45.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v45.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v22
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v46.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v46.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v47.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v47.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v48.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v48.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v22, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v22
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:336
	global_load_b128 v[19:22], v[31:32], off offset:320
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v49.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[42:49], v[19:26], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:368
	global_load_b128 v[42:45], v[31:32], off offset:352
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v15, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v16
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v17
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:400
	global_load_b128 v[42:45], v[31:32], off offset:384
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v23.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v23.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v24.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v24.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v25.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v25.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v26.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v26.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v11
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[19:26], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:432
	global_load_b128 v[42:45], v[31:32], off offset:416
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v11, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v12
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v12, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v13
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	s_clause 0x1
	global_load_b128 v[46:49], v[31:32], off offset:464
	global_load_b128 v[42:45], v[31:32], off offset:448
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v13, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v18.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v14
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v19.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 4, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v19.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 8, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v20.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 12, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v20.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 16, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v21.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 20, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v21.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v14, 24, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v22.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v22.h, v10.l, v9.l, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[15:22], v[42:49], v[1:8]
	global_load_b64 v[18:19], v[33:34], off offset:128
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v9, 15, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v11.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v11.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v12.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v12.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v13.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v13.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v18, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v14.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v18
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v14.h, v10.l, v9.l, v10.h
	v_and_b32_e32 v9, 15, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v15.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 4, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v15.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 8, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v16.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 12, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v16.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 16, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f16 v17.l, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 20, 4
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fma_f16 v17.h, v10.l, v9.l, v10.h
	v_bfe_u32 v9, v19, 24, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f16_f32_e32 v9.l, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f16 v18.l, v10.l, v9.l, v10.h
	v_lshrrev_b32_e32 v9, 28, v19
	s_clause 0x1
	global_load_b128 v[23:26], v[31:32], off offset:496
	global_load_b128 v[19:22], v[31:32], off offset:480
	v_cvt_f32_ubyte0_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9.l, v9
	v_fmac_f16_e32 v10.h, v10.l, v9.l
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mov_b16_e32 v18.h, v10.h
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[1:8], v[11:18], v[19:26], v[1:8]
	s_and_not1_b32 exec_lo, exec_lo, s5
	s_cbranch_execnz .LBB2_2
; %bb.3:                                ; %Flow255
	s_or_b32 exec_lo, exec_lo, s5
.LBB2_4:                                ; %Flow256
	v_and_b32_e32 v9, 31, v0
	s_mov_b32 s5, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v60, 2, v9
	v_lshl_or_b32 v9, v36, 10, v60
	ds_store_2addr_b32 v9, v1, v2 offset1:32
	ds_store_2addr_b32 v9, v3, v4 offset0:64 offset1:96
	ds_store_2addr_b32 v9, v5, v6 offset0:128 offset1:160
	ds_store_2addr_b32 v9, v7, v8 offset0:192 offset1:224
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v36
	s_cbranch_execz .LBB2_22
; %bb.5:                                ; %.preheader177
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_22
; %bb.6:                                ; %.preheader
	v_add_nc_u32_e32 v5, 0x1000, v60
	v_add_nc_u32_e32 v6, 0x1400, v60
	v_add_nc_u32_e32 v9, 0x80, v60
	v_add_nc_u32_e32 v1, 0x400, v60
	v_add_nc_u32_e32 v3, 0x800, v60
	v_add_nc_u32_e32 v4, 0xc00, v60
	v_add_nc_u32_e32 v10, 0x1800, v60
	v_add_nc_u32_e32 v17, 0x1c00, v60
	ds_load_2addr_b32 v[55:56], v5 offset0:32 offset1:64
	ds_load_2addr_b32 v[39:40], v5 offset0:96 offset1:128
	ds_load_2addr_b32 v[23:24], v5 offset0:160 offset1:192
	ds_load_2addr_b32 v[49:50], v6 offset0:64 offset1:96
	ds_load_2addr_b32 v[31:32], v6 offset0:128 offset1:160
	ds_load_2addr_b32 v[5:6], v6 offset0:192 offset1:224
	ds_load_2addr_b32 v[57:58], v10 offset0:32 offset1:64
	ds_load_2addr_b32 v[41:42], v10 offset0:96 offset1:128
	ds_load_2addr_b32 v[25:26], v10 offset0:160 offset1:192
	ds_load_2addr_stride64_b32 v[15:16], v9 offset0:3 offset1:4
	ds_load_2addr_stride64_b32 v[13:14], v9 offset0:11 offset1:12
	ds_load_2addr_stride64_b32 v[11:12], v9 offset0:19 offset1:20
	ds_load_2addr_stride64_b32 v[9:10], v9 offset0:27 offset1:28
	ds_load_2addr_b32 v[43:44], v1 offset0:64 offset1:96
	ds_load_2addr_b32 v[27:28], v1 offset0:128 offset1:160
	ds_load_2addr_b32 v[1:2], v1 offset0:192 offset1:224
	ds_load_2addr_b32 v[53:54], v3 offset0:32 offset1:64
	ds_load_2addr_b32 v[37:38], v3 offset0:96 offset1:128
	ds_load_2addr_b32 v[19:20], v3 offset0:160 offset1:192
	ds_load_2addr_b32 v[45:46], v4 offset0:64 offset1:96
	ds_load_2addr_b32 v[29:30], v4 offset0:128 offset1:160
	ds_load_2addr_b32 v[3:4], v4 offset0:192 offset1:224
	v_mad_i64_i32 v[21:22], null, s4, v35, 0
	ds_load_2addr_b32 v[51:52], v17 offset0:64 offset1:96
	ds_load_2addr_b32 v[35:36], v17 offset0:128 offset1:160
	ds_load_2addr_b32 v[17:18], v17 offset0:192 offset1:224
	ds_load_2addr_b32 v[47:48], v60 offset0:32 offset1:64
	ds_load_2addr_b32 v[33:34], v60 offset0:96 offset1:128
	ds_load_2addr_b32 v[7:8], v60 offset0:160 offset1:192
	v_lshrrev_b32_e32 v0, 4, v0
	s_mov_b32 s0, exec_lo
	v_lshlrev_b64 v[61:62], 2, v[21:22]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_or_b32_e32 v21, s1, v0
	v_add_co_u32 v0, vcc_lo, s2, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v59, null, s3, v62, vcc_lo
	v_ashrrev_i32_e32 v22, 31, v21
	v_cmpx_gt_i32_e64 s4, v21
	s_cbranch_execz .LBB2_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[61:62], 2, v[21:22]
	ds_load_2addr_stride64_b32 v[63:64], v60 offset1:4
	ds_load_2addr_stride64_b32 v[65:66], v60 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[67:68], v60 offset0:16 offset1:20
	v_add_co_u32 v61, vcc_lo, v0, v61
	v_add_co_ci_u32_e64 v62, null, v59, v62, vcc_lo
	global_load_b32 v69, v[61:62], off
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v63, 0, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v70, v63, v64
	ds_load_2addr_stride64_b32 v[63:64], v60 offset0:24 offset1:28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v60, v70, v65
	v_add_f32_e32 v60, v60, v66
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v67
	v_add_f32_e32 v60, v60, v68
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v63
	v_add_f32_e32 v60, v60, v64
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v69
	global_store_b32 v[61:62], v60, off
.LBB2_8:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v60, 2, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v60
	s_cbranch_execz .LBB2_10
; %bb.9:
	v_lshlrev_b64 v[60:61], 2, v[21:22]
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v47, 0, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v16, v47, v16
	v_add_co_u32 v60, vcc_lo, v0, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, v59, v61, vcc_lo
	v_add_f32_e32 v16, v16, v53
	global_load_b32 v62, v[60:61], off offset:8
	v_add_f32_e32 v14, v16, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v14, v55
	v_add_f32_e32 v12, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v57
	v_add_f32_e32 v10, v12, v10
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v10, v62
	global_store_b32 v[60:61], v10, off offset:8
.LBB2_10:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(15)
	v_or_b32_e32 v10, 4, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v10
	s_cbranch_execz .LBB2_12
; %bb.11:
	v_lshlrev_b64 v[60:61], 2, v[21:22]
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v12, 0, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v12, v43
	v_add_co_u32 v60, vcc_lo, v0, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, v59, v61, vcc_lo
	v_add_f32_e32 v12, v12, v54
	global_load_b32 v10, v[60:61], off offset:16
	v_add_f32_e32 v12, v12, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v56
	v_add_f32_e32 v12, v12, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v58
	v_add_f32_e32 v12, v12, v51
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v12, v10
	global_store_b32 v[60:61], v10, off offset:16
.LBB2_12:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 6, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v10
	s_cbranch_execz .LBB2_14
; %bb.13:
	s_waitcnt lgkmcnt(2)
	v_lshlrev_b64 v[47:48], 2, v[21:22]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v12, 0, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v12, v44
	v_add_co_u32 v47, vcc_lo, v0, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v48, null, v59, v48, vcc_lo
	v_add_f32_e32 v12, v12, v37
	global_load_b32 v10, v[47:48], off offset:24
	v_add_f32_e32 v12, v12, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v39
	v_add_f32_e32 v12, v12, v50
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v41
	v_add_f32_e32 v12, v12, v52
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v12, v10
	global_store_b32 v[47:48], v10, off offset:24
.LBB2_14:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 8, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v10
	s_cbranch_execz .LBB2_16
; %bb.15:
	s_waitcnt lgkmcnt(14)
	v_lshlrev_b64 v[43:44], 2, v[21:22]
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v12, 0, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v12, v27
	v_add_co_u32 v43, vcc_lo, v0, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v44, null, v59, v44, vcc_lo
	v_add_f32_e32 v12, v12, v38
	global_load_b32 v10, v[43:44], off offset:32
	v_add_f32_e32 v12, v12, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v40
	v_add_f32_e32 v12, v12, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v42
	v_add_f32_e32 v12, v12, v35
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v10, v12, v10
	global_store_b32 v[43:44], v10, off offset:32
.LBB2_16:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 10, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v10
	s_cbranch_execz .LBB2_18
; %bb.17:
	s_waitcnt lgkmcnt(1)
	v_lshlrev_b64 v[33:34], 2, v[21:22]
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, 0, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v7, v28
	v_add_co_u32 v33, vcc_lo, v0, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v34, null, v59, v34, vcc_lo
	v_add_f32_e32 v7, v7, v19
	global_load_b32 v10, v[33:34], off offset:40
	v_add_f32_e32 v7, v7, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v23
	v_add_f32_e32 v7, v7, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v25
	v_add_f32_e32 v7, v7, v36
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v7, v7, v10
	global_store_b32 v[33:34], v7, off offset:40
.LBB2_18:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(0)
	v_or_b32_e32 v7, 12, v21
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v7
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_lshlrev_b64 v[27:28], 2, v[21:22]
	v_add_f32_e32 v8, 0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v8, v1
	v_add_co_u32 v27, vcc_lo, v0, v27
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v28, null, v59, v28, vcc_lo
	v_add_f32_e32 v1, v1, v20
	global_load_b32 v7, v[27:28], off offset:48
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v24
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v26
	v_add_f32_e32 v1, v1, v17
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v7
	global_store_b32 v[27:28], v1, off offset:48
.LBB2_20:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v1, 14, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s4, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB2_22
; %bb.21:
	v_lshlrev_b64 v[7:8], 2, v[21:22]
	v_add_f32_e32 v5, 0, v15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v5, v2
	v_add_co_u32 v0, vcc_lo, v0, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, v59, v8, vcc_lo
	v_add_f32_e32 v2, v2, v13
	global_load_b32 v3, v[0:1], off offset:56
	v_add_f32_e32 v2, v2, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v11
	v_add_f32_e32 v2, v2, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v9
	v_add_f32_e32 v2, v2, v18
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v2, v2, v3
	global_store_b32 v[0:1], v2, off offset:56
.LBB2_22:                               ; %.loopexit
	s_endpgm
.Lfunc_end2:
	.size	gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds, .Lfunc_end2-gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 14
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
		.amdhsa_next_free_vgpr 71
		.amdhsa_next_free_sgpr 16
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds)<<4)&1008)>>4
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
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.num_vgpr, 71
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.numbered_sgpr, 16
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9620
; TotalNumSgprs: 18
; NumVgprs: 71
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 8192 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 71
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	gemm_mq4g256v2_residual_wmma_gfx1100_xlds ; -- Begin function gemm_mq4g256v2_residual_wmma_gfx1100_xlds
	.globl	gemm_mq4g256v2_residual_wmma_gfx1100_xlds
	.p2align	8
	.type	gemm_mq4g256v2_residual_wmma_gfx1100_xlds,@function
gemm_mq4g256v2_residual_wmma_gfx1100_xlds: ; @gemm_mq4g256v2_residual_wmma_gfx1100_xlds
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[4:7], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s7, s14, 4
	s_lshl_b32 s12, s15, 4
	s_cmp_ge_i32 s7, s4
	s_cselect_b32 s2, -1, 0
	s_cmp_ge_i32 s12, s6
	s_cselect_b32 s3, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s2, s2, s3
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB3_22
; %bb.1:
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x0
	s_load_b64 s[2:3], s[0:1], 0x10
	v_dual_mov_b32 v8, 0 :: v_dual_and_b32 v27, 15, v0
	v_lshrrev_b32_e32 v28, 5, v0
	v_lshrrev_b32_e32 v58, 4, v0
	s_cmpk_lt_i32 s5, 0x200
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v6, v8
	v_mov_b32_e32 v5, v8
	v_mov_b32_e32 v4, v8
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v2, v8
	v_mov_b32_e32 v1, v8
	s_cbranch_scc1 .LBB3_4
; %bb.2:                                ; %.lr.ph
	v_add_nc_u32_e32 v1, s12, v58
	v_bfe_u32 v7, v0, 5, 2
	v_or_b32_e32 v4, s7, v27
	s_ashr_i32 s1, s5, 31
	s_add_i32 s0, s4, -1
	v_cmp_gt_i32_e32 vcc_lo, s6, v1
	s_lshr_b32 s13, s1, 24
	s_lshr_b32 s1, s1, 23
	s_add_i32 s13, s5, s13
	v_lshrrev_b32_e32 v29, 7, v0
	v_dual_cndmask_b32 v2, 0, v1 :: v_dual_mov_b32 v1, 0
	v_lshlrev_b32_e32 v8, 6, v0
	v_lshlrev_b32_e32 v30, 5, v7
	v_cmp_gt_u32_e32 vcc_lo, 2, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_mad_u64_u32 v[5:6], null, v2, s5, 0
	v_ashrrev_i32_e32 v9, 31, v2
	v_lshlrev_b32_e32 v10, 10, v27
	v_lshlrev_b32_e32 v11, 7, v28
	s_add_i32 s1, s5, s1
	s_ashr_i32 s1, s1, 9
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_u64_u32 v[2:3], null, v9, s5, v[6:7]
	v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v12, 0x3c0, v8
	v_min_i32_e32 v9, s0, v4
	s_ashr_i32 s0, s13, 8
	v_mov_b32_e32 v4, v1
	s_mulk_i32 s0, 0x88
	v_lshl_or_b32 v32, v58, 10, v12
	v_mov_b32_e32 v6, v2
	s_waitcnt lgkmcnt(0)
	v_mad_i64_i32 v[25:26], null, s0, v9, s[8:9]
	v_mov_b32_e32 v2, v1
	s_mov_b32 s9, 0
	v_lshlrev_b64 v[7:8], 1, v[5:6]
	v_mov_b32_e32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_add_nc_u32 v31, v10, v11
	s_mov_b32 s8, s9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v7, s0, s10, v7
	v_add_co_ci_u32_e64 v8, null, s11, v8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v33, s0, v7, v12
	v_add_co_ci_u32_e64 v34, null, 0, v8, s0
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	s_lshl_b64 s[10:11], s[8:9], 1
	v_mad_u64_u32 v[35:36], null, 0x88, v29, v[25:26]
	v_add_co_u32 v21, s0, v33, s10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v22, null, s11, v34, s0
	v_add_nc_u32_e32 v29, 2, v29
	s_add_i32 s1, s1, -1
	s_addk_i32 s8, 0x200
	s_clause 0x3
	global_load_b128 v[9:12], v[21:22], off offset:48
	global_load_b128 v[13:16], v[21:22], off offset:32
	global_load_b128 v[17:20], v[21:22], off offset:16
	global_load_b128 v[21:24], v[21:22], off
	v_add_co_u32 v37, s0, v35, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v38, null, 0, v36, s0
	s_cmp_eq_u32 s1, 0
	s_waitcnt vmcnt(0)
	ds_store_b128 v32, v[21:24]
	ds_store_b128 v32, v[17:20] offset:16
	ds_store_b128 v32, v[13:16] offset:32
	ds_store_b128 v32, v[9:12] offset:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_clause 0x2
	global_load_b128 v[51:54], v[37:38], off offset:24
	global_load_b64 v[55:56], v[35:36], off
	global_load_b128 v[59:62], v[37:38], off offset:8
	ds_load_b128 v[17:20], v31
	ds_load_b128 v[21:24], v31 offset:16
	ds_load_b128 v[9:12], v31 offset:32
	ds_load_b128 v[13:16], v31 offset:48
	ds_load_b128 v[39:42], v31 offset:80
	ds_load_b128 v[35:38], v31 offset:64
	ds_load_b128 v[47:50], v31 offset:112
	ds_load_b128 v[43:46], v31 offset:96
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v88, 15, v51
	s_waitcnt vmcnt(0)
	v_dual_cndmask_b32 v91, v56, v55 :: v_dual_and_b32 v74, 15, v61
	v_and_b32_e32 v55, 15, v59
	v_bfe_u32 v56, v59, 4, 4
	v_bfe_u32 v57, v59, 8, 4
	v_bfe_u32 v63, v59, 12, 4
	v_bfe_u32 v64, v59, 16, 4
	v_bfe_u32 v65, v59, 20, 4
	v_bfe_u32 v66, v59, 24, 4
	v_lshrrev_b32_e32 v59, 28, v59
	v_and_b32_e32 v67, 15, v60
	v_bfe_u32 v68, v60, 4, 4
	v_bfe_u32 v69, v60, 8, 4
	v_bfe_u32 v70, v60, 12, 4
	v_bfe_u32 v71, v60, 16, 4
	v_bfe_u32 v72, v60, 20, 4
	v_bfe_u32 v73, v60, 24, 4
	v_lshrrev_b32_e32 v60, 28, v60
	v_bfe_u32 v75, v61, 4, 4
	v_bfe_u32 v76, v61, 8, 4
	v_bfe_u32 v77, v61, 12, 4
	v_bfe_u32 v78, v61, 16, 4
	v_bfe_u32 v79, v61, 20, 4
	v_bfe_u32 v80, v61, 24, 4
	v_lshrrev_b32_e32 v61, 28, v61
	v_and_b32_e32 v81, 15, v62
	v_bfe_u32 v82, v62, 4, 4
	v_bfe_u32 v83, v62, 8, 4
	v_bfe_u32 v84, v62, 12, 4
	v_bfe_u32 v85, v62, 16, 4
	v_bfe_u32 v86, v62, 20, 4
	v_bfe_u32 v87, v62, 24, 4
	v_lshrrev_b32_e32 v62, 28, v62
	v_bfe_u32 v89, v51, 4, 4
	v_bfe_u32 v90, v51, 8, 4
	v_bfe_u32 v92, v51, 12, 4
	v_bfe_u32 v93, v51, 16, 4
	v_bfe_u32 v94, v51, 20, 4
	v_bfe_u32 v95, v51, 24, 4
	v_lshrrev_b32_e32 v51, 28, v51
	v_and_b32_e32 v96, 15, v52
	v_bfe_u32 v97, v52, 4, 4
	v_bfe_u32 v98, v52, 8, 4
	v_bfe_u32 v99, v52, 12, 4
	v_bfe_u32 v100, v52, 16, 4
	v_bfe_u32 v101, v52, 20, 4
	v_bfe_u32 v102, v52, 24, 4
	v_lshrrev_b32_e32 v52, 28, v52
	v_and_b32_e32 v103, 15, v53
	v_bfe_u32 v104, v53, 4, 4
	v_bfe_u32 v105, v53, 8, 4
	v_bfe_u32 v106, v53, 12, 4
	v_bfe_u32 v107, v53, 16, 4
	v_bfe_u32 v108, v53, 20, 4
	v_bfe_u32 v109, v53, 24, 4
	v_lshrrev_b32_e32 v53, 28, v53
	v_and_b32_e32 v110, 15, v54
	v_bfe_u32 v111, v54, 4, 4
	v_bfe_u32 v112, v54, 8, 4
	v_bfe_u32 v113, v54, 12, 4
	v_bfe_u32 v114, v54, 16, 4
	v_bfe_u32 v115, v54, 20, 4
	v_bfe_u32 v116, v54, 24, 4
	v_lshrrev_b32_e32 v54, 28, v54
	v_cvt_f32_ubyte0_e32 v55, v55
	v_cvt_f32_ubyte0_e32 v56, v56
	v_cvt_f32_ubyte0_e32 v57, v57
	v_cvt_f32_ubyte0_e32 v63, v63
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v59, v59
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v73, v73
	v_cvt_f32_ubyte0_e32 v60, v60
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v79, v79
	v_cvt_f32_ubyte0_e32 v80, v80
	v_cvt_f32_ubyte0_e32 v61, v61
	v_cvt_f32_ubyte0_e32 v81, v81
	v_cvt_f32_ubyte0_e32 v82, v82
	v_cvt_f32_ubyte0_e32 v83, v83
	v_cvt_f32_ubyte0_e32 v84, v84
	v_cvt_f32_ubyte0_e32 v85, v85
	v_cvt_f32_ubyte0_e32 v86, v86
	v_cvt_f32_ubyte0_e32 v87, v87
	v_cvt_f32_ubyte0_e32 v62, v62
	v_cvt_f32_ubyte0_e32 v117, v51
	v_cvt_f32_ubyte0_e32 v118, v52
	v_cvt_f32_ubyte0_e32 v119, v53
	v_cvt_f32_ubyte0_e32 v120, v54
	v_cvt_f16_f32_e32 v51.l, v55
	v_cvt_f16_f32_e32 v51.h, v56
	v_cvt_f16_f32_e32 v52.l, v57
	v_cvt_f16_f32_e32 v52.h, v63
	v_cvt_f16_f32_e32 v53.l, v64
	v_cvt_f16_f32_e32 v53.h, v65
	v_cvt_f16_f32_e32 v54.l, v66
	v_cvt_f16_f32_e32 v54.h, v59
	v_cvt_f16_f32_e32 v55.l, v67
	v_cvt_f16_f32_e32 v55.h, v68
	v_cvt_f16_f32_e32 v56.l, v69
	v_cvt_f16_f32_e32 v56.h, v70
	v_cvt_f16_f32_e32 v57.l, v71
	v_cvt_f16_f32_e32 v57.h, v72
	v_cvt_f16_f32_e32 v66.l, v73
	v_cvt_f16_f32_e32 v66.h, v60
	v_cvt_f32_ubyte0_e32 v88, v88
	v_cvt_f32_ubyte0_e32 v89, v89
	v_cvt_f32_ubyte0_e32 v90, v90
	v_cvt_f32_ubyte0_e32 v92, v92
	v_cvt_f32_ubyte0_e32 v93, v93
	v_cvt_f32_ubyte0_e32 v94, v94
	v_cvt_f32_ubyte0_e32 v95, v95
	v_cvt_f32_ubyte0_e32 v96, v96
	v_cvt_f32_ubyte0_e32 v97, v97
	v_cvt_f32_ubyte0_e32 v98, v98
	v_cvt_f32_ubyte0_e32 v99, v99
	v_cvt_f32_ubyte0_e32 v100, v100
	v_cvt_f32_ubyte0_e32 v101, v101
	v_cvt_f32_ubyte0_e32 v102, v102
	v_cvt_f32_ubyte0_e32 v103, v103
	v_cvt_f32_ubyte0_e32 v104, v104
	v_cvt_f32_ubyte0_e32 v105, v105
	v_cvt_f32_ubyte0_e32 v106, v106
	v_cvt_f32_ubyte0_e32 v107, v107
	v_cvt_f32_ubyte0_e32 v108, v108
	v_cvt_f32_ubyte0_e32 v109, v109
	v_cvt_f32_ubyte0_e32 v110, v110
	v_cvt_f32_ubyte0_e32 v111, v111
	v_cvt_f32_ubyte0_e32 v112, v112
	v_cvt_f32_ubyte0_e32 v113, v113
	v_cvt_f32_ubyte0_e32 v114, v114
	v_cvt_f32_ubyte0_e32 v115, v115
	v_cvt_f32_ubyte0_e32 v116, v116
	v_cvt_f16_f32_e32 v67.l, v74
	v_cvt_f16_f32_e32 v67.h, v75
	v_cvt_f16_f32_e32 v68.l, v76
	v_cvt_f16_f32_e32 v68.h, v77
	v_cvt_f16_f32_e32 v69.l, v78
	v_cvt_f16_f32_e32 v69.h, v79
	v_cvt_f16_f32_e32 v70.l, v80
	v_cvt_f16_f32_e32 v70.h, v61
	v_cvt_f16_f32_e32 v71.l, v81
	v_cvt_f16_f32_e32 v71.h, v82
	v_cvt_f16_f32_e32 v72.l, v83
	v_cvt_f16_f32_e32 v72.h, v84
	v_cvt_f16_f32_e32 v73.l, v85
	v_cvt_f16_f32_e32 v73.h, v86
	v_cvt_f16_f32_e32 v74.l, v87
	v_cvt_f16_f32_e32 v74.h, v62
	v_fma_f16 v59.l, v91.l, v51.l, v91.h
	v_fma_f16 v59.h, v91.l, v51.h, v91.h
	v_fma_f16 v60.l, v91.l, v52.l, v91.h
	v_fma_f16 v60.h, v91.l, v52.h, v91.h
	v_fma_f16 v61.l, v91.l, v53.l, v91.h
	v_fma_f16 v61.h, v91.l, v53.h, v91.h
	v_fma_f16 v62.l, v91.l, v54.l, v91.h
	v_fma_f16 v62.h, v91.l, v54.h, v91.h
	v_fma_f16 v63.l, v91.l, v55.l, v91.h
	v_fma_f16 v63.h, v91.l, v55.h, v91.h
	v_fma_f16 v64.l, v91.l, v56.l, v91.h
	v_fma_f16 v64.h, v91.l, v56.h, v91.h
	v_fma_f16 v65.l, v91.l, v57.l, v91.h
	v_fma_f16 v65.h, v91.l, v57.h, v91.h
	v_fma_f16 v66.l, v91.l, v66.l, v91.h
	v_fma_f16 v66.h, v91.l, v66.h, v91.h
	v_cvt_f16_f32_e32 v75.l, v88
	v_cvt_f16_f32_e32 v75.h, v89
	v_cvt_f16_f32_e32 v76.l, v90
	v_cvt_f16_f32_e32 v76.h, v92
	v_cvt_f16_f32_e32 v77.l, v93
	v_cvt_f16_f32_e32 v77.h, v94
	v_cvt_f16_f32_e32 v78.l, v95
	v_cvt_f16_f32_e32 v78.h, v117
	v_cvt_f16_f32_e32 v79.l, v96
	v_cvt_f16_f32_e32 v79.h, v97
	v_cvt_f16_f32_e32 v80.l, v98
	v_cvt_f16_f32_e32 v80.h, v99
	v_cvt_f16_f32_e32 v81.l, v100
	v_cvt_f16_f32_e32 v81.h, v101
	v_cvt_f16_f32_e32 v82.l, v102
	v_cvt_f16_f32_e32 v82.h, v118
	v_cvt_f16_f32_e32 v83.l, v103
	v_cvt_f16_f32_e32 v83.h, v104
	v_cvt_f16_f32_e32 v84.l, v105
	v_cvt_f16_f32_e32 v84.h, v106
	v_cvt_f16_f32_e32 v85.l, v107
	v_cvt_f16_f32_e32 v85.h, v108
	v_cvt_f16_f32_e32 v86.l, v109
	v_cvt_f16_f32_e32 v86.h, v119
	v_cvt_f16_f32_e32 v87.l, v110
	v_cvt_f16_f32_e32 v87.h, v111
	v_cvt_f16_f32_e32 v88.l, v112
	v_cvt_f16_f32_e32 v88.h, v113
	v_cvt_f16_f32_e32 v89.l, v114
	v_cvt_f16_f32_e32 v89.h, v115
	v_cvt_f16_f32_e32 v90.l, v116
	v_cvt_f16_f32_e32 v90.h, v120
	v_fma_f16 v67.l, v91.l, v67.l, v91.h
	v_fma_f16 v67.h, v91.l, v67.h, v91.h
	v_fma_f16 v68.l, v91.l, v68.l, v91.h
	v_fma_f16 v68.h, v91.l, v68.h, v91.h
	v_fma_f16 v69.l, v91.l, v69.l, v91.h
	v_fma_f16 v69.h, v91.l, v69.h, v91.h
	v_fma_f16 v70.l, v91.l, v70.l, v91.h
	v_fma_f16 v70.h, v91.l, v70.h, v91.h
	v_fma_f16 v71.l, v91.l, v71.l, v91.h
	v_fma_f16 v71.h, v91.l, v71.h, v91.h
	v_fma_f16 v72.l, v91.l, v72.l, v91.h
	v_fma_f16 v72.h, v91.l, v72.h, v91.h
	v_fma_f16 v73.l, v91.l, v73.l, v91.h
	v_fma_f16 v73.h, v91.l, v73.h, v91.h
	v_fma_f16 v74.l, v91.l, v74.l, v91.h
	v_fma_f16 v74.h, v91.l, v74.h, v91.h
	s_waitcnt lgkmcnt(6)
	v_wmma_f32_16x16x16_f16 v[1:8], v[59:66], v[17:24], v[1:8]
	v_fma_f16 v75.l, v91.l, v75.l, v91.h
	v_fma_f16 v75.h, v91.l, v75.h, v91.h
	v_fma_f16 v76.l, v91.l, v76.l, v91.h
	v_fma_f16 v76.h, v91.l, v76.h, v91.h
	v_fma_f16 v77.l, v91.l, v77.l, v91.h
	v_fma_f16 v77.h, v91.l, v77.h, v91.h
	v_fma_f16 v78.l, v91.l, v78.l, v91.h
	v_fma_f16 v78.h, v91.l, v78.h, v91.h
	v_fma_f16 v79.l, v91.l, v79.l, v91.h
	v_fma_f16 v79.h, v91.l, v79.h, v91.h
	v_fma_f16 v80.l, v91.l, v80.l, v91.h
	v_fma_f16 v80.h, v91.l, v80.h, v91.h
	v_fma_f16 v81.l, v91.l, v81.l, v91.h
	v_fma_f16 v81.h, v91.l, v81.h, v91.h
	v_fma_f16 v82.l, v91.l, v82.l, v91.h
	v_fma_f16 v82.h, v91.l, v82.h, v91.h
	v_fma_f16 v83.l, v91.l, v83.l, v91.h
	v_fma_f16 v83.h, v91.l, v83.h, v91.h
	v_fma_f16 v84.l, v91.l, v84.l, v91.h
	v_fma_f16 v84.h, v91.l, v84.h, v91.h
	v_fma_f16 v85.l, v91.l, v85.l, v91.h
	v_fma_f16 v85.h, v91.l, v85.h, v91.h
	v_fma_f16 v86.l, v91.l, v86.l, v91.h
	v_fma_f16 v86.h, v91.l, v86.h, v91.h
	v_fma_f16 v87.l, v91.l, v87.l, v91.h
	v_fma_f16 v87.h, v91.l, v87.h, v91.h
	v_fma_f16 v88.l, v91.l, v88.l, v91.h
	v_fma_f16 v88.h, v91.l, v88.h, v91.h
	v_fma_f16 v89.l, v91.l, v89.l, v91.h
	v_fma_f16 v89.h, v91.l, v89.h, v91.h
	v_fma_f16 v90.l, v91.l, v90.l, v91.h
	v_fmac_f16_e32 v91.h, v91.l, v90.h
	s_waitcnt lgkmcnt(4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[67:74], v[9:16], v[1:8]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mov_b16_e32 v90.h, v91.h
	s_waitcnt lgkmcnt(2)
	v_wmma_f32_16x16x16_f16 v[1:8], v[75:82], v[35:42], v[1:8]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[83:90], v[43:50], v[1:8]
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc0 .LBB3_3
.LBB3_4:                                ; %Flow510
	v_and_b32_e32 v0, 31, v0
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v60, 2, v0
	v_lshl_or_b32 v0, v28, 10, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v0, 0x4000, v0
	ds_store_2addr_b32 v0, v1, v2 offset1:32
	ds_store_2addr_b32 v0, v3, v4 offset0:64 offset1:96
	ds_store_2addr_b32 v0, v5, v6 offset0:128 offset1:160
	ds_store_2addr_b32 v0, v7, v8 offset0:192 offset1:224
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v28
	s_cbranch_execz .LBB3_22
; %bb.5:                                ; %.preheader403
	v_or_b32_e32 v16, s12, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s6, v16
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB3_22
; %bb.6:                                ; %.preheader
	v_add_nc_u32_e32 v4, 0x5000, v60
	v_add_nc_u32_e32 v5, 0x5400, v60
	v_add_nc_u32_e32 v8, 0x80, v60
	v_add_nc_u32_e32 v0, 0x4000, v60
	v_add_nc_u32_e32 v1, 0x4400, v60
	v_add_nc_u32_e32 v2, 0x4800, v60
	v_add_nc_u32_e32 v3, 0x4c00, v60
	v_add_nc_u32_e32 v9, 0x5800, v60
	v_add_nc_u32_e32 v17, 0x5c00, v60
	ds_load_2addr_b32 v[54:55], v4 offset0:32 offset1:64
	ds_load_2addr_b32 v[38:39], v4 offset0:96 offset1:128
	ds_load_2addr_b32 v[22:23], v4 offset0:160 offset1:192
	ds_load_2addr_b32 v[46:47], v5 offset0:64 offset1:96
	ds_load_2addr_b32 v[30:31], v5 offset0:128 offset1:160
	ds_load_2addr_b32 v[4:5], v5 offset0:192 offset1:224
	ds_load_2addr_b32 v[56:57], v9 offset0:32 offset1:64
	ds_load_2addr_b32 v[40:41], v9 offset0:96 offset1:128
	ds_load_2addr_b32 v[24:25], v9 offset0:160 offset1:192
	ds_load_2addr_stride64_b32 v[14:15], v8 offset0:67 offset1:68
	ds_load_2addr_stride64_b32 v[12:13], v8 offset0:75 offset1:76
	ds_load_2addr_stride64_b32 v[10:11], v8 offset0:83 offset1:84
	ds_load_2addr_stride64_b32 v[8:9], v8 offset0:91 offset1:92
	ds_load_2addr_b32 v[48:49], v0 offset0:32 offset1:64
	ds_load_2addr_b32 v[32:33], v0 offset0:96 offset1:128
	ds_load_2addr_b32 v[6:7], v0 offset0:160 offset1:192
	ds_load_2addr_b32 v[42:43], v1 offset0:64 offset1:96
	ds_load_2addr_b32 v[26:27], v1 offset0:128 offset1:160
	ds_load_2addr_b32 v[0:1], v1 offset0:192 offset1:224
	ds_load_2addr_b32 v[52:53], v2 offset0:32 offset1:64
	ds_load_2addr_b32 v[36:37], v2 offset0:96 offset1:128
	ds_load_2addr_b32 v[18:19], v2 offset0:160 offset1:192
	ds_load_2addr_b32 v[44:45], v3 offset0:64 offset1:96
	ds_load_2addr_b32 v[28:29], v3 offset0:128 offset1:160
	ds_load_2addr_b32 v[2:3], v3 offset0:192 offset1:224
	v_mad_i64_i32 v[20:21], null, s4, v16, 0
	ds_load_2addr_b32 v[50:51], v17 offset0:64 offset1:96
	ds_load_2addr_b32 v[34:35], v17 offset0:128 offset1:160
	ds_load_2addr_b32 v[16:17], v17 offset0:192 offset1:224
	s_mov_b32 s0, exec_lo
	v_lshlrev_b64 v[61:62], 2, v[20:21]
	v_or_b32_e32 v20, s7, v58
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v58, vcc_lo, s2, v61
	v_add_co_ci_u32_e64 v59, null, s3, v62, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_ashrrev_i32_e32 v21, 31, v20
	v_cmpx_gt_i32_e64 s4, v20
	s_cbranch_execz .LBB3_8
; %bb.7:
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b64 v[61:62], 2, v[20:21]
	v_or_b32_e32 v60, 0x4000, v60
	ds_load_2addr_stride64_b32 v[63:64], v60 offset1:4
	ds_load_2addr_stride64_b32 v[65:66], v60 offset0:8 offset1:12
	ds_load_2addr_stride64_b32 v[67:68], v60 offset0:16 offset1:20
	v_add_co_u32 v61, vcc_lo, v58, v61
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v62, null, v59, v62, vcc_lo
	global_load_b32 v69, v[61:62], off
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v63, 0, v63
	v_add_f32_e32 v70, v63, v64
	ds_load_2addr_stride64_b32 v[63:64], v60 offset0:24 offset1:28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v60, v70, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v66
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v60, v60, v67
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v68
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v60, v60, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v60, v60, v64
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v60, v60, v69
	global_store_b32 v[61:62], v60, off
.LBB3_8:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v60, 2, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v60
	s_cbranch_execz .LBB3_10
; %bb.9:
	v_lshlrev_b64 v[60:61], 2, v[20:21]
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v48, 0, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v15, v48, v15
	v_add_co_u32 v60, vcc_lo, v58, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, v59, v61, vcc_lo
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v15, v15, v52
	global_load_b32 v62, v[60:61], off offset:8
	v_add_f32_e32 v13, v15, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v13, v13, v54
	v_add_f32_e32 v11, v13, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v56
	v_add_f32_e32 v9, v11, v9
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v9, v62
	global_store_b32 v[60:61], v9, off offset:8
.LBB3_10:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(15)
	v_or_b32_e32 v9, 4, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v9
	s_cbranch_execz .LBB3_12
; %bb.11:
	v_lshlrev_b64 v[60:61], 2, v[20:21]
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v11, 0, v49
	s_waitcnt lgkmcnt(11)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v11, v42
	v_add_co_u32 v60, vcc_lo, v58, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v61, null, v59, v61, vcc_lo
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v11, v11, v53
	global_load_b32 v9, v[60:61], off offset:16
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v11, v11, v44
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v55
	v_add_f32_e32 v11, v11, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v57
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v11, v11, v50
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v11, v9
	global_store_b32 v[60:61], v9, off offset:16
.LBB3_12:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v9, 6, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v9
	s_cbranch_execz .LBB3_14
; %bb.13:
	s_waitcnt lgkmcnt(14)
	v_lshlrev_b64 v[48:49], 2, v[20:21]
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v11, 0, v32
	s_waitcnt lgkmcnt(11)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v11, v43
	v_add_co_u32 v48, vcc_lo, v58, v48
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v49, null, v59, v49, vcc_lo
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v11, v11, v36
	global_load_b32 v9, v[48:49], off offset:24
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v11, v11, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v38
	v_add_f32_e32 v11, v11, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v40
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v11, v11, v51
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v11, v9
	global_store_b32 v[48:49], v9, off offset:24
.LBB3_14:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v9, 8, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v9
	s_cbranch_execz .LBB3_16
; %bb.15:
	s_waitcnt lgkmcnt(11)
	v_lshlrev_b64 v[42:43], 2, v[20:21]
	v_add_f32_e32 v11, 0, v33
	s_waitcnt lgkmcnt(10)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v11, v26
	v_add_co_u32 v42, vcc_lo, v58, v42
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v43, null, v59, v43, vcc_lo
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v11, v11, v37
	global_load_b32 v9, v[42:43], off offset:32
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v11, v11, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v39
	v_add_f32_e32 v11, v11, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v41
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v11, v11, v34
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v11, v9
	global_store_b32 v[42:43], v9, off offset:32
.LBB3_16:
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v9, 10, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v9
	s_cbranch_execz .LBB3_18
; %bb.17:
	s_waitcnt lgkmcnt(13)
	v_lshlrev_b64 v[32:33], 2, v[20:21]
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v6, 0, v6
	s_waitcnt lgkmcnt(10)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v6, v6, v27
	v_add_co_u32 v32, vcc_lo, v58, v32
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v33, null, v59, v33, vcc_lo
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v6, v6, v18
	global_load_b32 v9, v[32:33], off offset:40
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v6, v6, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v22
	v_add_f32_e32 v6, v6, v31
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v24
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v6, v6, v35
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v6, v6, v9
	global_store_b32 v[32:33], v6, off offset:40
.LBB3_18:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(12)
	v_or_b32_e32 v6, 12, v20
	s_mov_b32 s0, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_i32_e64 s4, v6
	s_cbranch_execz .LBB3_20
; %bb.19:
	s_waitcnt lgkmcnt(10)
	v_lshlrev_b64 v[26:27], 2, v[20:21]
	v_add_f32_e32 v7, 0, v7
	s_waitcnt lgkmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v0, v7, v0
	v_add_co_u32 v26, vcc_lo, v58, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v27, null, v59, v27, vcc_lo
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v0, v0, v19
	global_load_b32 v6, v[26:27], off offset:48
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v0, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v23
	v_add_f32_e32 v0, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v25
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v16
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v6
	global_store_b32 v[26:27], v0, off offset:48
.LBB3_20:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(9)
	v_or_b32_e32 v0, 14, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s4, v0
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB3_22
; %bb.21:
	v_lshlrev_b64 v[6:7], 2, v[20:21]
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v2, 0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v2, v1
	v_add_co_u32 v6, vcc_lo, v58, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v7, null, v59, v7, vcc_lo
	v_add_f32_e32 v1, v1, v12
	global_load_b32 v0, v[6:7], off offset:56
	v_add_f32_e32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v10
	v_add_f32_e32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v8
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v1, v17
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v0, v1, v0
	global_store_b32 v[6:7], v0, off offset:56
.LBB3_22:                               ; %.loopexit
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	gemm_mq4g256v2_residual_wmma_gfx1100_xlds, .Lfunc_end3-gemm_mq4g256v2_residual_wmma_gfx1100_xlds
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm_mq4g256v2_residual_wmma_gfx1100_xlds
		.amdhsa_group_segment_fixed_size 24576
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 14
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
		.amdhsa_next_free_vgpr 121
		.amdhsa_next_free_sgpr 16
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-gemm_mq4g256v2_residual_wmma_gfx1100_xlds)<<4)&1008)>>4
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
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.num_vgpr, 121
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.num_agpr, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.numbered_sgpr, 16
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.num_named_barrier, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.private_seg_size, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.uses_vcc, 1
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.uses_flat_scratch, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.has_dyn_sized_stack, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.has_recursion, 0
	.set .Lgemm_mq4g256v2_residual_wmma_gfx1100_xlds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3844
; TotalNumSgprs: 18
; NumVgprs: 121
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 24576 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 121
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
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
	.type	__hip_cuid_c805b471c36f01f0,@object ; @__hip_cuid_c805b471c36f01f0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c805b471c36f01f0
__hip_cuid_c805b471c36f01f0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c805b471c36f01f0, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c805b471c36f01f0
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
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     62
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     62
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     71
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
    .gfx1250_revision: B0
    .group_segment_fixed_size: 24576
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           gemm_mq4g256v2_residual_wmma_gfx1100_xlds
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         gemm_mq4g256v2_residual_wmma_gfx1100_xlds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     121
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
