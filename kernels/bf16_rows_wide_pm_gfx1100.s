.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected gemv_bf16_xf32_rows_x4_t8_pm_gfx1100
.globl gemv_bf16_xf32_rows_x4_t8_pm_gfx1100
.p2align 8
.type gemv_bf16_xf32_rows_x4_t8_pm_gfx1100,@function
gemv_bf16_xf32_rows_x4_t8_pm_gfx1100:
	s_load_b256 s[72:79], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s60, s14
	s_cmp_lt_u32 s60, s26
	s_cbranch_scc1 .Lbf16xtx_t8_x4_matrix0
	s_sub_u32 s60, s60, s26
	s_cmp_lt_u32 s60, s27
	s_cbranch_scc1 .Lbf16xtx_t8_x4_matrix1
	s_sub_u32 s60, s60, s27
	s_cmp_lt_u32 s60, s28
	s_cbranch_scc1 .Lbf16xtx_t8_x4_matrix2
	s_sub_u32 s60, s60, s28
	s_cmp_lt_u32 s60, s29
	s_cbranch_scc1 .Lbf16xtx_t8_x4_matrix3
	s_sub_u32 s60, s60, s29
	s_branch .Lbf16xtx_t8_x4_end
	.Lbf16xtx_t8_x4_matrix0:
	s_mov_b32 s4, s72
	s_mov_b32 s5, s73
	s_mov_b32 s32, s18
	s_mov_b32 s33, s19
	s_mov_b32 s34, s26
	s_branch .Lbf16xtx_t8_x4_selected
	.Lbf16xtx_t8_x4_matrix1:
	s_mov_b32 s4, s74
	s_mov_b32 s5, s75
	s_mov_b32 s32, s20
	s_mov_b32 s33, s21
	s_mov_b32 s34, s27
	s_branch .Lbf16xtx_t8_x4_selected
	.Lbf16xtx_t8_x4_matrix2:
	s_mov_b32 s4, s76
	s_mov_b32 s5, s77
	s_mov_b32 s32, s22
	s_mov_b32 s33, s23
	s_mov_b32 s34, s28
	s_branch .Lbf16xtx_t8_x4_selected
	.Lbf16xtx_t8_x4_matrix3:
	s_mov_b32 s4, s78
	s_mov_b32 s5, s79
	s_mov_b32 s32, s24
	s_mov_b32 s33, s25
	s_mov_b32 s34, s29
	s_branch .Lbf16xtx_t8_x4_selected
	.Lbf16xtx_t8_x4_selected:
	s_mov_b32 s43, 0
	s_cmp_ge_u32 s60, s34
	s_cbranch_scc1 .Lbf16xtx_t8_x4_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_end
	s_cmp_gt_u32 s31, 64
	s_cbranch_scc1 .Lbf16xtx_t8_x4_end
	s_cmp_le_i32 s30, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_end
	s_lshl_b32 s41, s15, 3
	s_cmp_ge_u32 s41, s31
	s_cbranch_scc1 .Lbf16xtx_t8_x4_end
	s_sub_u32 s45, s31, s41
	s_min_u32 s40, s45, 8
	s_mov_b32 s42, s30
	s_lshl_b32 s53, s30, 2
	s_lshl_b32 s54, s34, 2
	s_lshl_b32 s45, s30, 1
	s_mul_i32 s46, s60, s45
	s_mul_hi_u32 s47, s60, s45
	s_add_u32 s4, s4, s46
	s_addc_u32 s5, s5, s47
	s_mul_i32 s57, s43, s42
	s_lshl_b32 s45, s57, 1
	s_add_u32 s4, s4, s45
	s_addc_u32 s5, s5, 0
	s_lshl_b32 s57, s57, 2
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_and_b32 s58, s42, 7
	v_and_b32_e32 v5, 31, v0
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_x4_initialized
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	.Lbf16xtx_t8_x4_initialized:
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar
	s_add_i32 s59, s42, 0xff
	s_lshr_b32 s59, s59, 8
	s_mov_b32 s52, 0
	.Lbf16xtx_t8_x4_fast_loop:
	s_lshl_b32 s45, s52, 8
	s_sub_u32 s49, s42, s45
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	s_lshl_b32 s50, s45, 1
	s_lshl_b32 s51, s45, 2
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v40, v24, v16
	v_fmac_f32_e32 v41, v25, v17
	v_fmac_f32_e32 v42, v26, v18
	v_fmac_f32_e32 v43, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	v_fmac_f32_e32 v45, v25, v17
	v_fmac_f32_e32 v46, v26, v18
	v_fmac_f32_e32 v47, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	v_fmac_f32_e32 v49, v25, v17
	v_fmac_f32_e32 v50, v26, v18
	v_fmac_f32_e32 v51, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	v_fmac_f32_e32 v53, v25, v17
	v_fmac_f32_e32 v54, v26, v18
	v_fmac_f32_e32 v55, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	v_fmac_f32_e32 v57, v25, v17
	v_fmac_f32_e32 v58, v26, v18
	v_fmac_f32_e32 v59, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	v_fmac_f32_e32 v61, v25, v17
	v_fmac_f32_e32 v62, v26, v18
	v_fmac_f32_e32 v63, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	v_fmac_f32_e32 v65, v25, v17
	v_fmac_f32_e32 v66, v26, v18
	v_fmac_f32_e32 v67, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_advanced
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	v_fmac_f32_e32 v69, v25, v17
	v_fmac_f32_e32 v70, v26, v18
	v_fmac_f32_e32 v71, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_x4_fast_advanced:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s59
	s_cbranch_scc1 .Lbf16xtx_t8_x4_fast_loop
	s_branch .Lbf16xtx_t8_x4_reduce
	.Lbf16xtx_t8_x4_scalar:
	s_add_i32 s59, s42, 31
	s_lshr_b32 s59, s59, 5
	s_mov_b32 s52, 0
	.Lbf16xtx_t8_x4_scalar_loop:
	s_lshl_b32 s45, s52, 5
	s_sub_u32 s49, s42, s45
	s_mov_b32 s48, s49
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 32
	s_cselect_b32 s48, -1, s48
	s_lshl_b32 s50, s45, 1
	s_lshl_b32 s51, s45, 2
	v_lshlrev_b32_e32 v1, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_mov_b32 exec_lo, s48
	buffer_load_u16 v16, v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v40, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_advanced
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_x4_scalar_advanced:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s59
	s_cbranch_scc1 .Lbf16xtx_t8_x4_scalar_loop
	.Lbf16xtx_t8_x4_reduce:
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v43, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	ds_swizzle_b32 v24, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v47, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	ds_swizzle_b32 v24, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v51, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	ds_swizzle_b32 v24, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v55, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	ds_swizzle_b32 v24, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v59, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	ds_swizzle_b32 v24, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v63, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	ds_swizzle_b32 v24, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v67, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_x4_reduced
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	ds_swizzle_b32 v24, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v71, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_x4_reduced:
	.Lbf16xtx_t8_x4_end:
	s_endpgm
.Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1100_end:
.size gemv_bf16_xf32_rows_x4_t8_pm_gfx1100, .Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1100_end-gemv_bf16_xf32_rows_x4_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_rows_x4_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 96
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
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_rows_x4_t8_pm_gfx1100_end-gemv_bf16_xf32_rows_x4_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.text
.protected gemv_bf16_xf32_k4_rows_t8_pm_gfx1100
.globl gemv_bf16_xf32_k4_rows_t8_pm_gfx1100
.p2align 8
.type gemv_bf16_xf32_k4_rows_t8_pm_gfx1100,@function
gemv_bf16_xf32_k4_rows_t8_pm_gfx1100:
	s_load_b256 s[72:79], s[0:1], null
	s_load_b128 s[16:19], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s44, s16
	s_mov_b32 s61, s17
	s_mov_b32 s31, s18
	s_mov_b32 s16, s74
	s_mov_b32 s17, s75
	s_mov_b32 s32, s76
	s_mov_b32 s33, s77
	s_mov_b32 s34, s78
	s_mov_b32 s30, s79
	s_mov_b32 s60, s14
	v_lshrrev_b32_e32 v13, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s43, v13
	s_mov_b32 s4, s72
	s_mov_b32 s5, s73
	s_cmp_ge_u32 s60, s34
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_cmp_eq_u32 s31, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_cmp_gt_u32 s31, 64
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_cmp_le_i32 s30, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_lshl_b32 s41, s15, 3
	s_cmp_ge_u32 s41, s31
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_sub_u32 s45, s31, s41
	s_min_u32 s40, s45, 8
	s_and_b32 s45, s30, 31
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_lshr_b32 s42, s30, 2
	s_lshl_b32 s53, s30, 2
	s_lshl_b32 s54, s34, 2
	s_lshl_b32 s45, s30, 1
	s_mul_i32 s46, s60, s45
	s_mul_hi_u32 s47, s60, s45
	s_add_u32 s4, s4, s46
	s_addc_u32 s5, s5, s47
	s_mul_i32 s57, s43, s42
	s_lshl_b32 s45, s57, 1
	s_add_u32 s4, s4, s45
	s_addc_u32 s5, s5, 0
	s_lshl_b32 s57, s57, 2
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_and_b32 s58, s42, 7
	v_and_b32_e32 v5, 31, v0
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_hc_initialized
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	.Lbf16xtx_t8_hc_initialized:
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar
	s_add_i32 s59, s42, 0xff
	s_lshr_b32 s59, s59, 8
	s_mov_b32 s52, 0
	.Lbf16xtx_t8_hc_fast_loop:
	s_lshl_b32 s45, s52, 8
	s_sub_u32 s49, s42, s45
	s_lshr_b32 s48, s49, 3
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 0x100
	s_cselect_b32 s48, -1, s48
	s_lshl_b32 s50, s45, 1
	s_lshl_b32 s51, s45, 2
	v_lshlrev_b32_e32 v1, 4, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_mov_b32 exec_lo, s48
	buffer_load_b128 v[16:19], v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v19
	v_lshlrev_b32_e32 v22, 16, v19
	v_and_b32_e32 v21, 0xffff0000, v18
	v_lshlrev_b32_e32 v20, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v17
	v_lshlrev_b32_e32 v18, 16, v17
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v40, v24, v16
	v_fmac_f32_e32 v41, v25, v17
	v_fmac_f32_e32 v42, v26, v18
	v_fmac_f32_e32 v43, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v40, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v41, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v42, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v43, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	v_fmac_f32_e32 v45, v25, v17
	v_fmac_f32_e32 v46, v26, v18
	v_fmac_f32_e32 v47, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v44, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v45, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v46, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v47, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	v_fmac_f32_e32 v49, v25, v17
	v_fmac_f32_e32 v50, v26, v18
	v_fmac_f32_e32 v51, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v48, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v49, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v50, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v51, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	v_fmac_f32_e32 v53, v25, v17
	v_fmac_f32_e32 v54, v26, v18
	v_fmac_f32_e32 v55, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v52, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v53, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v54, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v55, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	v_fmac_f32_e32 v57, v25, v17
	v_fmac_f32_e32 v58, v26, v18
	v_fmac_f32_e32 v59, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v56, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v57, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v58, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v59, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	v_fmac_f32_e32 v61, v25, v17
	v_fmac_f32_e32 v62, v26, v18
	v_fmac_f32_e32 v63, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v60, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v61, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v62, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v63, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	v_fmac_f32_e32 v65, v25, v17
	v_fmac_f32_e32 v66, v26, v18
	v_fmac_f32_e32 v67, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v64, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v65, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v66, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v67, v31, v23
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_advanced
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 5, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	global_load_b32 v25, v3, s[16:17] offset:4
	global_load_b32 v26, v3, s[16:17] offset:8
	global_load_b32 v27, v3, s[16:17] offset:12
	global_load_b32 v28, v3, s[16:17] offset:16
	global_load_b32 v29, v3, s[16:17] offset:20
	global_load_b32 v30, v3, s[16:17] offset:24
	global_load_b32 v31, v3, s[16:17] offset:28
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	v_fmac_f32_e32 v69, v25, v17
	v_fmac_f32_e32 v70, v26, v18
	v_fmac_f32_e32 v71, v27, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v68, v28, v20
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v69, v29, v21
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v70, v30, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v71, v31, v23
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_hc_fast_advanced:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s59
	s_cbranch_scc1 .Lbf16xtx_t8_hc_fast_loop
	s_branch .Lbf16xtx_t8_hc_reduce
	.Lbf16xtx_t8_hc_scalar:
	s_add_i32 s59, s42, 31
	s_lshr_b32 s59, s59, 5
	s_mov_b32 s52, 0
	.Lbf16xtx_t8_hc_scalar_loop:
	s_lshl_b32 s45, s52, 5
	s_sub_u32 s49, s42, s45
	s_mov_b32 s48, s49
	s_lshl_b32 s48, -1, s48
	s_not_b32 s48, s48
	s_cmp_ge_u32 s49, 32
	s_cselect_b32 s48, -1, s48
	s_lshl_b32 s50, s45, 1
	s_lshl_b32 s51, s45, 2
	v_lshlrev_b32_e32 v1, 1, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s50, v1
	s_mov_b32 exec_lo, s48
	buffer_load_u16 v16, v1, s[4:7], 0 offen
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 16, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v40, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v44, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v48, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v52, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v56, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v60, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v64, v24, v16
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_advanced
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s53
	s_add_i32 s45, s45, s57
	s_add_i32 s45, s45, s51
	v_lshlrev_b32_e32 v3, 2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s45, v3
	s_mov_b32 exec_lo, s48
	global_load_b32 v24, v3, s[16:17]
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v68, v24, v16
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_hc_scalar_advanced:
	s_add_i32 s52, s52, 1
	s_cmp_lt_u32 s52, s59
	s_cbranch_scc1 .Lbf16xtx_t8_hc_scalar_loop
	.Lbf16xtx_t8_hc_reduce:
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v41, v41, v40
	v_add_f32_e32 v43, v43, v42
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v41
	ds_swizzle_b32 v24, v43 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v43
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v43, v43, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v43
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v45, v45, v44
	v_add_f32_e32 v47, v47, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v45
	ds_swizzle_b32 v24, v47 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v47
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v47, v47, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v47 offset:16
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v49, v49, v48
	v_add_f32_e32 v51, v51, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v49
	ds_swizzle_b32 v24, v51 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v51
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v51, v51, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v51 offset:32
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v53, v53, v52
	v_add_f32_e32 v55, v55, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v53
	ds_swizzle_b32 v24, v55 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v55
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v55, v55, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v55 offset:48
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v57, v57, v56
	v_add_f32_e32 v59, v59, v58
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v57
	ds_swizzle_b32 v24, v59 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v59
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v59, v59, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v59 offset:64
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v61, v61, v60
	v_add_f32_e32 v63, v63, v62
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v61
	ds_swizzle_b32 v24, v63 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v63
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v63, v63, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v63 offset:80
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v65, v65, v64
	v_add_f32_e32 v67, v67, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v65
	ds_swizzle_b32 v24, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v67
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v67 offset:96
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_hc_reduced
	v_add_f32_e32 v69, v69, v68
	v_add_f32_e32 v71, v71, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v69
	ds_swizzle_b32 v24, v71 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 24, v5
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 28, v5
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 30, v5
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	v_cmp_gt_u32_e32 vcc_lo, 31, v5
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v5, 2
	ds_bpermute_b32 v24, v4, v71
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v71, v71, v24
	s_lshl_b32 s45, s43, 2
	v_mov_b32_e32 v9, s45
	s_mov_b32 exec_lo, 1
	ds_store_b32 v9, v71 offset:112
	s_waitcnt lgkmcnt(0)
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_hc_reduced:
	s_barrier
	s_cmp_lg_u32 s43, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	s_cmp_le_u32 s40, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 0
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act0_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn0_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn0_0_mask
	.Lbf16xtx_t8_hc_rn0_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn0_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn0_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn0_1_mask
	.Lbf16xtx_t8_hc_rn0_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn0_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn0_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn0_2_mask
	.Lbf16xtx_t8_hc_rn0_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn0_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act0_done:
	s_add_i32 s45, s41, 0
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 1
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 16
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act1_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn1_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn1_0_mask
	.Lbf16xtx_t8_hc_rn1_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn1_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn1_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn1_1_mask
	.Lbf16xtx_t8_hc_rn1_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn1_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn1_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn1_2_mask
	.Lbf16xtx_t8_hc_rn1_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn1_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act1_done:
	s_add_i32 s45, s41, 1
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 2
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 32
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act2_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn2_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn2_0_mask
	.Lbf16xtx_t8_hc_rn2_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn2_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn2_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn2_1_mask
	.Lbf16xtx_t8_hc_rn2_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn2_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn2_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn2_2_mask
	.Lbf16xtx_t8_hc_rn2_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn2_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act2_done:
	s_add_i32 s45, s41, 2
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 3
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 48
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act3_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn3_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn3_0_mask
	.Lbf16xtx_t8_hc_rn3_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn3_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn3_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn3_1_mask
	.Lbf16xtx_t8_hc_rn3_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn3_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn3_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn3_2_mask
	.Lbf16xtx_t8_hc_rn3_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn3_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act3_done:
	s_add_i32 s45, s41, 3
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 4
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 64
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act4_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn4_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn4_0_mask
	.Lbf16xtx_t8_hc_rn4_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn4_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn4_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn4_1_mask
	.Lbf16xtx_t8_hc_rn4_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn4_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn4_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn4_2_mask
	.Lbf16xtx_t8_hc_rn4_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn4_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act4_done:
	s_add_i32 s45, s41, 4
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 5
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 0x50
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act5_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn5_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn5_0_mask
	.Lbf16xtx_t8_hc_rn5_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn5_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn5_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn5_1_mask
	.Lbf16xtx_t8_hc_rn5_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn5_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn5_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn5_2_mask
	.Lbf16xtx_t8_hc_rn5_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn5_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act5_done:
	s_add_i32 s45, s41, 5
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 6
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 0x60
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act6_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn6_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn6_0_mask
	.Lbf16xtx_t8_hc_rn6_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn6_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn6_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn6_1_mask
	.Lbf16xtx_t8_hc_rn6_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn6_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn6_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn6_2_mask
	.Lbf16xtx_t8_hc_rn6_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn6_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act6_done:
	s_add_i32 s45, s41, 6
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	s_cmp_le_u32 s40, 7
	s_cbranch_scc1 .Lbf16xtx_t8_hc_end
	v_mov_b32_e32 v9, 0x70
	ds_load_b128 v[16:19], v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v16, v16, v17
	v_add_f32_e32 v18, v18, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v16, v16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v24, v16
	s_cmp_eq_u32 s61, 0
	s_cbranch_scc1 .Lbf16xtx_t8_hc_act7_done
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn7_0_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn7_0_mask
	.Lbf16xtx_t8_hc_rn7_0_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn7_0_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, s44, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn7_1_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn7_1_mask
	.Lbf16xtx_t8_hc_rn7_1_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn7_1_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mul_f32_e64 v24, 0xbfb8aa3b, s64
	v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64
	s_delay_alu instid0(VALU_DEP_2)
	v_fma_f32 v25, 0xbfb8aa3b, s64, -v24
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v26, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_fmamk_f32 v25, s64, 0xb2a5705f, v25
	s_delay_alu instid0(VALU_DEP_2)
	v_sub_f32_e32 v24, v24, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_i32_f32_e32 v25, v26
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v24, v24
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0, v24, vcc_lo
	v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v24, 1.0, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v26, v25
	s_waitcnt_depctr depctr_va_vdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v25, v26, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v26, v27, v26
	v_div_scale_f32 v27, vcc_lo, s64, v24, s64
	s_delay_alu instid0(VALU_DEP_1) | instid1(TRANS32_DEP_1)
	v_mul_f32_e32 v28, v27, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v29, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v28, v29, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v25, -v25, v28, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v25, v25, v26, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v24, v25, v24, s64
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s64, v24
	s_and_b32 s65, s64, 0x7f800000
	s_cmp_eq_u32 s65, 0x7f800000
	s_cbranch_scc1 .Lbf16xtx_t8_hc_rn7_2_special
	s_bfe_u32 s65, s64, 0x10010
	s_add_u32 s65, s64, s65
	s_add_u32 s65, s65, 0x7fff
	s_branch .Lbf16xtx_t8_hc_rn7_2_mask
	.Lbf16xtx_t8_hc_rn7_2_special:
	s_and_b32 s66, s64, 0xffff
	s_or_b32 s67, s64, 0x10000
	s_cmp_eq_u32 s66, 0
	s_cselect_b32 s65, s64, s67
	.Lbf16xtx_t8_hc_rn7_2_mask:
	s_and_b32 s64, s65, 0xffff0000
	v_mov_b32_e32 v24, s64
	.Lbf16xtx_t8_hc_act7_done:
	s_add_i32 s45, s41, 7
	s_mul_i32 s45, s45, s54
	s_lshl_b32 s55, s60, 2
	s_add_i32 s45, s45, s55
	v_mov_b32_e32 v3, s45
	s_mov_b32 exec_lo, 1
	global_store_b32 v3, v24, s[32:33]
	s_waitcnt_vscnt null, 0x0
	s_mov_b32 exec_lo, -1
	.Lbf16xtx_t8_hc_end:
	s_endpgm
.Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1100_end:
.size gemv_bf16_xf32_k4_rows_t8_pm_gfx1100, .Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1100_end-gemv_bf16_xf32_k4_rows_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_bf16_xf32_k4_rows_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 128
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 48
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
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 80
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_bf16_xf32_k4_rows_t8_pm_gfx1100_end-gemv_bf16_xf32_k4_rows_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A2
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A3
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y0
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y1
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y2
        .offset: 56
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y3
        .offset: 64
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M0
        .offset: 72
        .size: 4
        .value_kind: by_value
      - 
        .name: M1
        .offset: 76
        .size: 4
        .value_kind: by_value
      - 
        .name: M2
        .offset: 80
        .size: 4
        .value_kind: by_value
      - 
        .name: M3
        .offset: 84
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 88
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 92
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .max_flat_workgroup_size: 32
    .name: gemv_bf16_xf32_rows_x4_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 82
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_rows_x4_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: W
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: X
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: Y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: scale
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: hc_act
        .offset: 36
        .size: 4
        .value_kind: by_value
      - 
        .name: R
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: gemv_bf16_xf32_k4_rows_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 82
    .sgpr_spill_count: 0
    .symbol: gemv_bf16_xf32_k4_rows_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
