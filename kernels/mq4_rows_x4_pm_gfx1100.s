.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_rows_x4_t8_pm_gfx1100
.globl gemv_mq4g256v2_rows_x4_t8_pm_gfx1100
.p2align 8
.type gemv_mq4g256v2_rows_x4_t8_pm_gfx1100,@function
gemv_mq4g256v2_rows_x4_t8_pm_gfx1100:
	s_load_b256 s[80:87], s[0:1], null
	s_load_b256 s[16:23], s[0:1], 0x20
	s_load_b128 s[24:27], s[0:1], 0x40
	s_load_b128 s[28:31], s[0:1], 0x50
	s_mov_b32 s40, s14
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s57, s31
	s_cmp_eq_u32 s57, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_gt_u32 s57, 64
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_le_i32 s30, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_and_b32 s45, s30, 0xff
	s_cmp_lg_u32 s45, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_lt_i32 s26, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_lt_i32 s27, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_lt_i32 s28, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_lt_i32 s29, 0
	s_cbranch_scc1 .Lmq4rxt8_end
	s_mov_b32 s56, s15
	s_cmp_ge_u32 s56, 8
	s_cbranch_scc1 .Lmq4rxt8_end
	s_lshl_b32 s56, s56, 3
	s_cmp_ge_u32 s56, s57
	s_cbranch_scc1 .Lmq4rxt8_end
	s_sub_i32 s31, s57, s56
	s_cmp_gt_u32 s31, 8
	s_cselect_b32 s31, 8, s31
	s_add_i32 s36, s26, 1
	s_lshr_b32 s36, s36, 1
	s_add_i32 s45, s27, 1
	s_lshr_b32 s45, s45, 1
	s_add_i32 s37, s36, s45
	s_add_i32 s45, s28, 1
	s_lshr_b32 s45, s45, 1
	s_add_i32 s38, s37, s45
	s_add_i32 s45, s29, 1
	s_lshr_b32 s45, s45, 1
	s_add_i32 s39, s38, s45
	s_cmp_ge_u32 s40, s39
	s_cbranch_scc1 .Lmq4rxt8_end
	s_cmp_lt_u32 s40, s38
	s_cselect_b64 s[4:5], s[84:85], s[86:87]
	s_cselect_b64 s[32:33], s[22:23], s[24:25]
	s_cselect_b32 s34, s28, s29
	s_cselect_b32 s35, s37, s38
	s_cmp_lt_u32 s40, s37
	s_cselect_b64 s[4:5], s[82:83], s[4:5]
	s_cselect_b64 s[32:33], s[20:21], s[32:33]
	s_cselect_b32 s34, s27, s34
	s_cselect_b32 s35, s36, s35
	s_cmp_lt_u32 s40, s36
	s_cselect_b64 s[4:5], s[80:81], s[4:5]
	s_cselect_b64 s[32:33], s[18:19], s[32:33]
	s_cselect_b32 s34, s26, s34
	s_cselect_b32 s35, 0, s35
	s_sub_i32 s41, s40, s35
	s_lshl_b32 s41, s41, 1
	s_lshr_b32 s43, s30, 8
	s_mul_i32 s44, s43, 0x88
	s_add_i32 s42, s41, 1
	s_cmp_lt_i32 s42, s34
	s_cselect_b32 s42, s42, s41
	s_mul_i32 s46, s41, s44
	s_mul_i32 s47, s42, s44
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_lshl_b32 s52, s30, 2
	s_lshl_b32 s53, s34, 2
	s_lshl_b32 s54, s41, 2
	s_and_b32 s59, s43, -4
	s_mov_b32 s58, 1
	s_mul_i32 s55, s56, s52
	s_add_u32 s16, s16, s55
	s_addc_u32 s17, s17, 0
	s_mul_i32 s55, s56, s53
	s_add_u32 s32, s32, s55
	s_addc_u32 s33, s33, 0
	s_mul_i32 s64, s52, 0
	s_add_u32 s64, s16, s64
	s_addc_u32 s65, s17, 0
	s_mul_i32 s66, s52, 1
	s_add_u32 s66, s16, s66
	s_addc_u32 s67, s17, 0
	s_mul_i32 s68, s52, 2
	s_add_u32 s68, s16, s68
	s_addc_u32 s69, s17, 0
	s_mul_i32 s70, s52, 3
	s_add_u32 s70, s16, s70
	s_addc_u32 s71, s17, 0
	s_mul_i32 s72, s52, 4
	s_add_u32 s72, s16, s72
	s_addc_u32 s73, s17, 0
	s_mul_i32 s74, s52, 5
	s_add_u32 s74, s16, s74
	s_addc_u32 s75, s17, 0
	s_mul_i32 s76, s52, 6
	s_add_u32 s76, s16, s76
	s_addc_u32 s77, s17, 0
	s_mul_i32 s78, s52, 7
	s_add_u32 s78, s16, s78
	s_addc_u32 s79, s17, 0
	s_cmp_lg_u32 s58, 0
	s_cbranch_scc1 .Lmq4rxt8_scalar_dispatch
	.Lmq4rxt8_scalar_dispatch:
	s_cmp_eq_u32 s31, 1
	s_cbranch_scc1 .Lmq4rxt8_b9_start
	s_cmp_eq_u32 s31, 2
	s_cbranch_scc1 .Lmq4rxt8_b10_start
	s_cmp_eq_u32 s31, 3
	s_cbranch_scc1 .Lmq4rxt8_b11_start
	s_cmp_eq_u32 s31, 4
	s_cbranch_scc1 .Lmq4rxt8_b12_start
	s_cmp_eq_u32 s31, 5
	s_cbranch_scc1 .Lmq4rxt8_b13_start
	s_cmp_eq_u32 s31, 6
	s_cbranch_scc1 .Lmq4rxt8_b14_start
	s_cmp_eq_u32 s31, 7
	s_cbranch_scc1 .Lmq4rxt8_b15_start
	s_cmp_eq_u32 s31, 8
	s_cbranch_scc1 .Lmq4rxt8_b16_start
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b9_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b9_fold
	.Lmq4rxt8_b9_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	s_waitcnt vmcnt(2)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b9_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b9_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b9_s2
	s_branch .Lmq4rxt8_b9_s3
	.Lmq4rxt8_b9_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b9_s0_tail
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	s_branch .Lmq4rxt8_b9_s0_dot_done
	.Lmq4rxt8_b9_s0_tail:
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	.Lmq4rxt8_b9_s0_dot_done:
	s_branch .Lmq4rxt8_b9_group_done
	.Lmq4rxt8_b9_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b9_s1_tail
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	s_branch .Lmq4rxt8_b9_s1_dot_done
	.Lmq4rxt8_b9_s1_tail:
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	.Lmq4rxt8_b9_s1_dot_done:
	s_branch .Lmq4rxt8_b9_group_done
	.Lmq4rxt8_b9_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b9_s2_tail
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	s_branch .Lmq4rxt8_b9_s2_dot_done
	.Lmq4rxt8_b9_s2_tail:
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	.Lmq4rxt8_b9_s2_dot_done:
	s_branch .Lmq4rxt8_b9_group_done
	.Lmq4rxt8_b9_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b9_s3_tail
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	s_branch .Lmq4rxt8_b9_s3_dot_done
	.Lmq4rxt8_b9_s3_tail:
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	.Lmq4rxt8_b9_s3_dot_done:
	.Lmq4rxt8_b9_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b9_group
	.Lmq4rxt8_b9_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v41, v41, v17
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	global_store_b32 v8, v40, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b9_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	.Lmq4rxt8_b9_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b10_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b10_fold
	.Lmq4rxt8_b10_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b10_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b10_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b10_s2
	s_branch .Lmq4rxt8_b10_s3
	.Lmq4rxt8_b10_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b10_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	s_branch .Lmq4rxt8_b10_s0_dot_done
	.Lmq4rxt8_b10_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	.Lmq4rxt8_b10_s0_dot_done:
	s_branch .Lmq4rxt8_b10_group_done
	.Lmq4rxt8_b10_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b10_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	s_branch .Lmq4rxt8_b10_s1_dot_done
	.Lmq4rxt8_b10_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	.Lmq4rxt8_b10_s1_dot_done:
	s_branch .Lmq4rxt8_b10_group_done
	.Lmq4rxt8_b10_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b10_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	s_branch .Lmq4rxt8_b10_s2_dot_done
	.Lmq4rxt8_b10_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	.Lmq4rxt8_b10_s2_dot_done:
	s_branch .Lmq4rxt8_b10_group_done
	.Lmq4rxt8_b10_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b10_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	s_branch .Lmq4rxt8_b10_s3_dot_done
	.Lmq4rxt8_b10_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	.Lmq4rxt8_b10_s3_dot_done:
	.Lmq4rxt8_b10_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b10_group
	.Lmq4rxt8_b10_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v49, v49, v19
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b10_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	.Lmq4rxt8_b10_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b11_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b11_fold
	.Lmq4rxt8_b11_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b11_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b11_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b11_s2
	s_branch .Lmq4rxt8_b11_s3
	.Lmq4rxt8_b11_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b11_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	s_branch .Lmq4rxt8_b11_s0_dot_done
	.Lmq4rxt8_b11_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	.Lmq4rxt8_b11_s0_dot_done:
	s_branch .Lmq4rxt8_b11_group_done
	.Lmq4rxt8_b11_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b11_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	s_branch .Lmq4rxt8_b11_s1_dot_done
	.Lmq4rxt8_b11_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	.Lmq4rxt8_b11_s1_dot_done:
	s_branch .Lmq4rxt8_b11_group_done
	.Lmq4rxt8_b11_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b11_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	s_branch .Lmq4rxt8_b11_s2_dot_done
	.Lmq4rxt8_b11_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	.Lmq4rxt8_b11_s2_dot_done:
	s_branch .Lmq4rxt8_b11_group_done
	.Lmq4rxt8_b11_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b11_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	s_branch .Lmq4rxt8_b11_s3_dot_done
	.Lmq4rxt8_b11_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	.Lmq4rxt8_b11_s3_dot_done:
	.Lmq4rxt8_b11_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b11_group
	.Lmq4rxt8_b11_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v57, v57, v21
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b11_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	.Lmq4rxt8_b11_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b12_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b12_fold
	.Lmq4rxt8_b12_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b12_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b12_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b12_s2
	s_branch .Lmq4rxt8_b12_s3
	.Lmq4rxt8_b12_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b12_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	s_branch .Lmq4rxt8_b12_s0_dot_done
	.Lmq4rxt8_b12_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	.Lmq4rxt8_b12_s0_dot_done:
	s_branch .Lmq4rxt8_b12_group_done
	.Lmq4rxt8_b12_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b12_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	s_branch .Lmq4rxt8_b12_s1_dot_done
	.Lmq4rxt8_b12_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	.Lmq4rxt8_b12_s1_dot_done:
	s_branch .Lmq4rxt8_b12_group_done
	.Lmq4rxt8_b12_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b12_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	s_branch .Lmq4rxt8_b12_s2_dot_done
	.Lmq4rxt8_b12_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	.Lmq4rxt8_b12_s2_dot_done:
	s_branch .Lmq4rxt8_b12_group_done
	.Lmq4rxt8_b12_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b12_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	s_branch .Lmq4rxt8_b12_s3_dot_done
	.Lmq4rxt8_b12_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	.Lmq4rxt8_b12_s3_dot_done:
	.Lmq4rxt8_b12_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b12_group
	.Lmq4rxt8_b12_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v70, v70, v68 :: v_dual_add_f32 v71, v71, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v70 :: v_dual_add_f32 v65, v65, v71
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v65, v65, v23
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	s_mul_i32 s55, s53, 3
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v11, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	global_store_b32 v11, v64, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b12_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	global_store_b32 v11, v65, s[32:33] offset:4
	.Lmq4rxt8_b12_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b13_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b13_fold
	.Lmq4rxt8_b13_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b13_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b13_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b13_s2
	s_branch .Lmq4rxt8_b13_s3
	.Lmq4rxt8_b13_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b13_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	s_branch .Lmq4rxt8_b13_s0_dot_done
	.Lmq4rxt8_b13_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	.Lmq4rxt8_b13_s0_dot_done:
	s_branch .Lmq4rxt8_b13_group_done
	.Lmq4rxt8_b13_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b13_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	s_branch .Lmq4rxt8_b13_s1_dot_done
	.Lmq4rxt8_b13_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	.Lmq4rxt8_b13_s1_dot_done:
	s_branch .Lmq4rxt8_b13_group_done
	.Lmq4rxt8_b13_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b13_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	s_branch .Lmq4rxt8_b13_s2_dot_done
	.Lmq4rxt8_b13_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	.Lmq4rxt8_b13_s2_dot_done:
	s_branch .Lmq4rxt8_b13_group_done
	.Lmq4rxt8_b13_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b13_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	s_branch .Lmq4rxt8_b13_s3_dot_done
	.Lmq4rxt8_b13_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	.Lmq4rxt8_b13_s3_dot_done:
	.Lmq4rxt8_b13_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b13_group
	.Lmq4rxt8_b13_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v70, v70, v68 :: v_dual_add_f32 v71, v71, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v70 :: v_dual_add_f32 v65, v65, v71
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v78, v78, v76 :: v_dual_add_f32 v79, v79, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v78 :: v_dual_add_f32 v73, v73, v79
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v72 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v73 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v73, v73, v25
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	s_mul_i32 s55, s53, 3
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v11, s55
	s_mul_i32 s55, s53, 4
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v12, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	global_store_b32 v11, v64, s[32:33]
	global_store_b32 v12, v72, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b13_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	global_store_b32 v11, v65, s[32:33] offset:4
	global_store_b32 v12, v73, s[32:33] offset:4
	.Lmq4rxt8_b13_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b14_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b14_fold
	.Lmq4rxt8_b14_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b14_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b14_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b14_s2
	s_branch .Lmq4rxt8_b14_s3
	.Lmq4rxt8_b14_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b14_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	s_branch .Lmq4rxt8_b14_s0_dot_done
	.Lmq4rxt8_b14_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	.Lmq4rxt8_b14_s0_dot_done:
	s_branch .Lmq4rxt8_b14_group_done
	.Lmq4rxt8_b14_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b14_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	s_branch .Lmq4rxt8_b14_s1_dot_done
	.Lmq4rxt8_b14_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	.Lmq4rxt8_b14_s1_dot_done:
	s_branch .Lmq4rxt8_b14_group_done
	.Lmq4rxt8_b14_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b14_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	s_branch .Lmq4rxt8_b14_s2_dot_done
	.Lmq4rxt8_b14_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	.Lmq4rxt8_b14_s2_dot_done:
	s_branch .Lmq4rxt8_b14_group_done
	.Lmq4rxt8_b14_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b14_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	s_branch .Lmq4rxt8_b14_s3_dot_done
	.Lmq4rxt8_b14_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	.Lmq4rxt8_b14_s3_dot_done:
	.Lmq4rxt8_b14_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b14_group
	.Lmq4rxt8_b14_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v70, v70, v68 :: v_dual_add_f32 v71, v71, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v70 :: v_dual_add_f32 v65, v65, v71
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v78, v78, v76 :: v_dual_add_f32 v79, v79, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v78 :: v_dual_add_f32 v73, v73, v79
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v86, v86, v84 :: v_dual_add_f32 v87, v87, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v86 :: v_dual_add_f32 v81, v81, v87
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v72 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v73 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v80 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v81 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v81, v81, v27
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	s_mul_i32 s55, s53, 3
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v11, s55
	s_mul_i32 s55, s53, 4
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v12, s55
	s_mul_i32 s55, s53, 5
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v13, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	global_store_b32 v11, v64, s[32:33]
	global_store_b32 v12, v72, s[32:33]
	global_store_b32 v13, v80, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b14_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	global_store_b32 v11, v65, s[32:33] offset:4
	global_store_b32 v12, v73, s[32:33] offset:4
	global_store_b32 v13, v81, s[32:33] offset:4
	.Lmq4rxt8_b14_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b15_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b15_fold
	.Lmq4rxt8_b15_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b15_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b15_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b15_s2
	s_branch .Lmq4rxt8_b15_s3
	.Lmq4rxt8_b15_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b15_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v24 :: v_dual_add_f32 v89, v89, v25
	s_branch .Lmq4rxt8_b15_s0_dot_done
	.Lmq4rxt8_b15_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v24 :: v_dual_add_f32 v89, v89, v25
	.Lmq4rxt8_b15_s0_dot_done:
	s_branch .Lmq4rxt8_b15_group_done
	.Lmq4rxt8_b15_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b15_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v90, v24 :: v_dual_add_f32 v91, v91, v25
	s_branch .Lmq4rxt8_b15_s1_dot_done
	.Lmq4rxt8_b15_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v90, v24 :: v_dual_add_f32 v91, v91, v25
	.Lmq4rxt8_b15_s1_dot_done:
	s_branch .Lmq4rxt8_b15_group_done
	.Lmq4rxt8_b15_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b15_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v92, v24 :: v_dual_add_f32 v93, v93, v25
	s_branch .Lmq4rxt8_b15_s2_dot_done
	.Lmq4rxt8_b15_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v92, v24 :: v_dual_add_f32 v93, v93, v25
	.Lmq4rxt8_b15_s2_dot_done:
	s_branch .Lmq4rxt8_b15_group_done
	.Lmq4rxt8_b15_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b15_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v94, v24 :: v_dual_add_f32 v95, v95, v25
	s_branch .Lmq4rxt8_b15_s3_dot_done
	.Lmq4rxt8_b15_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v94, v24 :: v_dual_add_f32 v95, v95, v25
	.Lmq4rxt8_b15_s3_dot_done:
	.Lmq4rxt8_b15_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b15_group
	.Lmq4rxt8_b15_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v70, v70, v68 :: v_dual_add_f32 v71, v71, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v70 :: v_dual_add_f32 v65, v65, v71
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v78, v78, v76 :: v_dual_add_f32 v79, v79, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v78 :: v_dual_add_f32 v73, v73, v79
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v86, v86, v84 :: v_dual_add_f32 v87, v87, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v86 :: v_dual_add_f32 v81, v81, v87
	v_dual_add_f32 v88, v88, v90 :: v_dual_add_f32 v89, v89, v91
	v_dual_add_f32 v94, v94, v92 :: v_dual_add_f32 v95, v95, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v94 :: v_dual_add_f32 v89, v89, v95
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v72 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v73 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v80 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v81 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v89, v89, v29
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	s_mul_i32 s55, s53, 3
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v11, s55
	s_mul_i32 s55, s53, 4
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v12, s55
	s_mul_i32 s55, s53, 5
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v13, s55
	s_mul_i32 s55, s53, 6
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v14, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	global_store_b32 v11, v64, s[32:33]
	global_store_b32 v12, v72, s[32:33]
	global_store_b32 v13, v80, s[32:33]
	global_store_b32 v14, v88, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b15_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	global_store_b32 v11, v65, s[32:33] offset:4
	global_store_b32 v12, v73, s[32:33] offset:4
	global_store_b32 v13, v81, s[32:33] offset:4
	global_store_b32 v14, v89, s[32:33] offset:4
	.Lmq4rxt8_b15_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_b16_start:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s50, 0
	s_cmp_eq_u32 s43, 0
	s_cbranch_scc1 .Lmq4rxt8_b16_fold
	.Lmq4rxt8_b16_group:
	s_mul_i32 s48, s50, 0x88
	s_lshl_b32 s49, s50, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s48, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s48, v2
	buffer_load_b32 v8, v1, s[4:7], s46 offen
	buffer_load_b32 v10, v2, s[4:7], s46 offen offset:8
	buffer_load_b32 v9, v1, s[4:7], s47 offen
	buffer_load_b32 v11, v2, s[4:7], s47 offen offset:8
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s49, v3
	global_load_b128 v[105:108], v3, s[64:65]
	global_load_b128 v[109:112], v3, s[64:65] offset:16
	global_load_b128 v[113:116], v3, s[66:67]
	global_load_b128 v[117:120], v3, s[66:67] offset:16
	s_waitcnt vmcnt(4)
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v26, v11, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v26, v9, v26, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v10, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v17, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v17, v8, v17, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v27, v11, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v27, v9, v27, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v10, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v18, v8, v18, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v28, v11, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v28, v9, v28, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v10, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v19, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v19, v8, v19, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v29, v11, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v29, v9, v29, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v10, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v20, v8, v20, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v30, v11, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v30, v9, v30, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v10, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v21, v8, v21, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v31, v11, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v31, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v31, v9, v31, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v10, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v22, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v22, v8, v22, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v11, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v32, v9, v32, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v10, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v23, v8, v23, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v33, v11, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v33, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v33, v9, v33, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_and_b32 s51, s50, 3
	s_cmp_eq_u32 s51, 0
	s_cbranch_scc1 .Lmq4rxt8_b16_s0
	s_cmp_eq_u32 s51, 1
	s_cbranch_scc1 .Lmq4rxt8_b16_s1
	s_cmp_eq_u32 s51, 2
	s_cbranch_scc1 .Lmq4rxt8_b16_s2
	s_branch .Lmq4rxt8_b16_s3
	.Lmq4rxt8_b16_s0:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b16_s0_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v24 :: v_dual_add_f32 v89, v89, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v96, v96, v24 :: v_dual_add_f32 v97, v97, v25
	s_branch .Lmq4rxt8_b16_s0_dot_done
	.Lmq4rxt8_b16_s0_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v24 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v24 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v24 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v24 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v24 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v24 :: v_dual_add_f32 v81, v81, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v24 :: v_dual_add_f32 v89, v89, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v96, v96, v24 :: v_dual_add_f32 v97, v97, v25
	.Lmq4rxt8_b16_s0_dot_done:
	s_branch .Lmq4rxt8_b16_group_done
	.Lmq4rxt8_b16_s1:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b16_s1_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v90, v24 :: v_dual_add_f32 v91, v91, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v98, v98, v24 :: v_dual_add_f32 v99, v99, v25
	s_branch .Lmq4rxt8_b16_s1_dot_done
	.Lmq4rxt8_b16_s1_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v42, v24 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v50, v24 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v58, v24 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v66, v24 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v74, v24 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v82, v24 :: v_dual_add_f32 v83, v83, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v90, v24 :: v_dual_add_f32 v91, v91, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v98, v98, v24 :: v_dual_add_f32 v99, v99, v25
	.Lmq4rxt8_b16_s1_dot_done:
	s_branch .Lmq4rxt8_b16_group_done
	.Lmq4rxt8_b16_s2:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b16_s2_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v92, v24 :: v_dual_add_f32 v93, v93, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v100, v100, v24 :: v_dual_add_f32 v101, v101, v25
	s_branch .Lmq4rxt8_b16_s2_dot_done
	.Lmq4rxt8_b16_s2_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v44, v24 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v52, v24 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v60, v24 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v68, v24 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v76, v24 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v84, v24 :: v_dual_add_f32 v85, v85, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v92, v24 :: v_dual_add_f32 v93, v93, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v100, v100, v24 :: v_dual_add_f32 v101, v101, v25
	.Lmq4rxt8_b16_s2_dot_done:
	s_branch .Lmq4rxt8_b16_group_done
	.Lmq4rxt8_b16_s3:
	s_cmp_ge_u32 s50, s59
	s_cbranch_scc1 .Lmq4rxt8_b16_s3_tail
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v17, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v112
	v_mul_f32_e32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v94, v24 :: v_dual_add_f32 v95, v95, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v17, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v16, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v19, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v20, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v21, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v22, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v23, v120
	v_mul_f32_e32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v102, v102, v24 :: v_dual_add_f32 v103, v103, v25
	s_branch .Lmq4rxt8_b16_s3_dot_done
	.Lmq4rxt8_b16_s3_tail:
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v46, v24 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[68:69]
	global_load_b128 v[109:112], v3, s[68:69] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v54, v24 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[70:71]
	global_load_b128 v[117:120], v3, s[70:71] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v62, v24 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[72:73]
	global_load_b128 v[109:112], v3, s[72:73] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v70, v24 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[74:75]
	global_load_b128 v[117:120], v3, s[74:75] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v78, v24 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[76:77]
	global_load_b128 v[109:112], v3, s[76:77] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v86, v24 :: v_dual_add_f32 v87, v87, v25
	global_load_b128 v[113:116], v3, s[78:79]
	global_load_b128 v[117:120], v3, s[78:79] offset:16
	s_waitcnt vmcnt(2)
	v_mul_f32_e32 v24, v106, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v105, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v107, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v108, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v109, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v110, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v111, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v112, v23
	v_mul_f32_e32 v25, v106, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v105, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v107, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v108, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v109, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v110, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v111, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v112, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v94, v24 :: v_dual_add_f32 v95, v95, v25
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v24, v114, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v113, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v115, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v116, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v117, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v118, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v119, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v24, v120, v23
	v_mul_f32_e32 v25, v114, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v113, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v115, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v116, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v117, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v118, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v119, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v120, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v102, v102, v24 :: v_dual_add_f32 v103, v103, v25
	.Lmq4rxt8_b16_s3_dot_done:
	.Lmq4rxt8_b16_group_done:
	s_add_i32 s50, s50, 1
	s_cmp_lt_u32 s50, s43
	s_cbranch_scc1 .Lmq4rxt8_b16_group
	.Lmq4rxt8_b16_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v46, v46, v44 :: v_dual_add_f32 v47, v47, v45
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v46 :: v_dual_add_f32 v41, v41, v47
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v54, v54, v52 :: v_dual_add_f32 v55, v55, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v54 :: v_dual_add_f32 v49, v49, v55
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v62, v62, v60 :: v_dual_add_f32 v63, v63, v61
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v62 :: v_dual_add_f32 v57, v57, v63
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v70, v70, v68 :: v_dual_add_f32 v71, v71, v69
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v70 :: v_dual_add_f32 v65, v65, v71
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v78, v78, v76 :: v_dual_add_f32 v79, v79, v77
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v78 :: v_dual_add_f32 v73, v73, v79
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v86, v86, v84 :: v_dual_add_f32 v87, v87, v85
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v86 :: v_dual_add_f32 v81, v81, v87
	v_dual_add_f32 v88, v88, v90 :: v_dual_add_f32 v89, v89, v91
	v_dual_add_f32 v94, v94, v92 :: v_dual_add_f32 v95, v95, v93
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v94 :: v_dual_add_f32 v89, v89, v95
	v_dual_add_f32 v96, v96, v98 :: v_dual_add_f32 v97, v97, v99
	v_dual_add_f32 v102, v102, v100 :: v_dual_add_f32 v103, v103, v101
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v96, v96, v102 :: v_dual_add_f32 v97, v97, v103
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v72 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v73 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v80 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v81 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v96 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v97 offset:swizzle(BITMASK_PERM,"1pppp")
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v89, v89, v29
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v96, v96, v30
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	ds_bpermute_b32 v30, v4, v96
	ds_bpermute_b32 v31, v4, v97
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v89, v89, v29
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v96, v96, v30
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	ds_bpermute_b32 v30, v4, v96
	ds_bpermute_b32 v31, v4, v97
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v89, v89, v29
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v96, v96, v30
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	ds_bpermute_b32 v30, v4, v96
	ds_bpermute_b32 v31, v4, v97
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v89, v89, v29
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v96, v96, v30
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	ds_bpermute_b32 v22, v4, v64
	ds_bpermute_b32 v23, v4, v65
	ds_bpermute_b32 v24, v4, v72
	ds_bpermute_b32 v25, v4, v73
	ds_bpermute_b32 v26, v4, v80
	ds_bpermute_b32 v27, v4, v81
	ds_bpermute_b32 v28, v4, v88
	ds_bpermute_b32 v29, v4, v89
	ds_bpermute_b32 v30, v4, v96
	ds_bpermute_b32 v31, v4, v97
	s_waitcnt lgkmcnt(15)
	v_add_f32_e32 v40, v40, v16
	s_waitcnt lgkmcnt(14)
	v_add_f32_e32 v41, v41, v17
	s_waitcnt lgkmcnt(13)
	v_add_f32_e32 v48, v48, v18
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v49, v19
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v56, v56, v20
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v57, v57, v21
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v64, v64, v22
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v23
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v72, v72, v24
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v73, v73, v25
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v26
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v27
	s_waitcnt lgkmcnt(3)
	v_add_f32_e32 v88, v88, v28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v89, v89, v29
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v96, v96, v30
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v97, v97, v31
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s55, s53, 0
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v8, s55
	s_mul_i32 s55, s53, 1
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v9, s55
	s_mul_i32 s55, s53, 2
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v10, s55
	s_mul_i32 s55, s53, 3
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v11, s55
	s_mul_i32 s55, s53, 4
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v12, s55
	s_mul_i32 s55, s53, 5
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v13, s55
	s_mul_i32 s55, s53, 6
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v14, s55
	s_mul_i32 s55, s53, 7
	s_add_i32 s55, s55, s54
	v_mov_b32_e32 v15, s55
	global_store_b32 v8, v40, s[32:33]
	global_store_b32 v9, v48, s[32:33]
	global_store_b32 v10, v56, s[32:33]
	global_store_b32 v11, v64, s[32:33]
	global_store_b32 v12, v72, s[32:33]
	global_store_b32 v13, v80, s[32:33]
	global_store_b32 v14, v88, s[32:33]
	global_store_b32 v15, v96, s[32:33]
	s_cmp_eq_u32 s42, s41
	s_cbranch_scc1 .Lmq4rxt8_b16_stored
	global_store_b32 v8, v41, s[32:33] offset:4
	global_store_b32 v9, v49, s[32:33] offset:4
	global_store_b32 v10, v57, s[32:33] offset:4
	global_store_b32 v11, v65, s[32:33] offset:4
	global_store_b32 v12, v73, s[32:33] offset:4
	global_store_b32 v13, v81, s[32:33] offset:4
	global_store_b32 v14, v89, s[32:33] offset:4
	global_store_b32 v15, v97, s[32:33] offset:4
	.Lmq4rxt8_b16_stored:
	s_waitcnt_vscnt null, 0x0
	s_branch .Lmq4rxt8_end
	.Lmq4rxt8_end:
	s_endpgm
.Lgemv_mq4g256v2_rows_x4_t8_pm_gfx1100_end:
.size gemv_mq4g256v2_rows_x4_t8_pm_gfx1100, .Lgemv_mq4g256v2_rows_x4_t8_pm_gfx1100_end-gemv_mq4g256v2_rows_x4_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_rows_x4_t8_pm_gfx1100
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
	.amdhsa_next_free_vgpr 121
	.amdhsa_next_free_sgpr 88
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_rows_x4_t8_pm_gfx1100_end-gemv_mq4g256v2_rows_x4_t8_pm_gfx1100)<<4)&1008)>>4
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
    .name: gemv_mq4g256v2_rows_x4_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 90
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_rows_x4_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 121
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
