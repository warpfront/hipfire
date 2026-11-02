.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_multirow_r2_xbatch_pm
.globl gemv_mq4g256v2_multirow_r2_xbatch_pm
.p2align 8
.type gemv_mq4g256v2_multirow_r2_xbatch_pm,@function
gemv_mq4g256v2_multirow_r2_xbatch_pm:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b32 s3, s[0:1], 0x20
	s_lshl_b32 s12, ttmp9, 1
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lmx_end
	s_lshr_b32 s14, s11, 8
	s_mul_i32 s15, s14, 0x88
	s_add_co_i32 s13, s12, 1
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_mul_i32 s16, s12, s15
	s_mul_i32 s17, s13, s15
	s_mov_b32 s20, s4
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s22, -1
	s_mov_b32 s23, 0x31004000
	s_lshl_b32 s44, s11, 2
	s_lshl_b32 s45, s10, 2
	s_lshl_b32 s46, s12, 2
	s_mul_i32 s28, s44, 0
	s_mov_b32 s29, 0
	s_add_nc_u64 s[28:29], s[6:7], s[28:29]
	s_mul_i32 s30, s44, 1
	s_mov_b32 s31, 0
	s_add_nc_u64 s[30:31], s[6:7], s[30:31]
	s_mul_i32 s32, s44, 2
	s_mov_b32 s33, 0
	s_add_nc_u64 s[32:33], s[6:7], s[32:33]
	s_mul_i32 s34, s44, 3
	s_mov_b32 s35, 0
	s_add_nc_u64 s[34:35], s[6:7], s[34:35]
	s_mul_i32 s36, s44, 4
	s_mov_b32 s37, 0
	s_add_nc_u64 s[36:37], s[6:7], s[36:37]
	s_mul_i32 s38, s44, 5
	s_mov_b32 s39, 0
	s_add_nc_u64 s[38:39], s[6:7], s[38:39]
	s_mul_i32 s40, s44, 6
	s_mov_b32 s41, 0
	s_add_nc_u64 s[40:41], s[6:7], s[40:41]
	s_mul_i32 s42, s44, 7
	s_mov_b32 s43, 0
	s_add_nc_u64 s[42:43], s[6:7], s[42:43]
	s_cmp_eq_u32 s3, 1
	s_cbranch_scc1 .Lmx_b1
	s_cmp_eq_u32 s3, 2
	s_cbranch_scc1 .Lmx_b2
	s_cmp_eq_u32 s3, 3
	s_cbranch_scc1 .Lmx_b3
	s_cmp_eq_u32 s3, 4
	s_cbranch_scc1 .Lmx_b4
	s_cmp_eq_u32 s3, 5
	s_cbranch_scc1 .Lmx_b5
	s_cmp_eq_u32 s3, 6
	s_cbranch_scc1 .Lmx_b6
	s_cmp_eq_u32 s3, 7
	s_cbranch_scc1 .Lmx_b7
	s_cmp_eq_u32 s3, 8
	s_cbranch_scc1 .Lmx_b8
	s_branch .Lmx_end
	.Lmx_b1:
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b1_fold
	.Lmx_b1_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	s_wait_loadcnt 0x4
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x2
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b1_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b1_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b1_s2
	s_branch .Lmx_b1_s3
	.Lmx_b1_s0:
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	s_branch .Lmx_b1_group_done
	.Lmx_b1_s1:
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	s_branch .Lmx_b1_group_done
	.Lmx_b1_s2:
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	s_branch .Lmx_b1_group_done
	.Lmx_b1_s3:
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	.Lmx_b1_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b1_group
	.Lmx_b1_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v41, v41, v17
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x0
	v_add_f32_e32 v41, v41, v17
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_add_co_i32 s47, s47, s46
	v_mov_b32_e32 v8, s47
	global_store_b32 v8, v40, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b1_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	.Lmx_b1_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b2:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b2_fold
	.Lmx_b2_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b2_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b2_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b2_s2
	s_branch .Lmx_b2_s3
	.Lmx_b2_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	s_branch .Lmx_b2_group_done
	.Lmx_b2_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	s_branch .Lmx_b2_group_done
	.Lmx_b2_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	s_branch .Lmx_b2_group_done
	.Lmx_b2_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	.Lmx_b2_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b2_group
	.Lmx_b2_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v49, v49, v19
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	s_wait_dscnt 0x3
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v49, v49, v19
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b2_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	.Lmx_b2_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b3:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b3_fold
	.Lmx_b3_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b3_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b3_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b3_s2
	s_branch .Lmx_b3_s3
	.Lmx_b3_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	s_branch .Lmx_b3_group_done
	.Lmx_b3_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	s_branch .Lmx_b3_group_done
	.Lmx_b3_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	s_branch .Lmx_b3_group_done
	.Lmx_b3_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	.Lmx_b3_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b3_group
	.Lmx_b3_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x5
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x1
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x0
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_wait_dscnt 0x5
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x1
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x0
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_wait_dscnt 0x5
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x1
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x0
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_wait_dscnt 0x5
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x1
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x0
	v_add_f32_e32 v57, v57, v21
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v16, v4, v40
	ds_bpermute_b32 v17, v4, v41
	ds_bpermute_b32 v18, v4, v48
	ds_bpermute_b32 v19, v4, v49
	ds_bpermute_b32 v20, v4, v56
	ds_bpermute_b32 v21, v4, v57
	s_wait_dscnt 0x5
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x3
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x2
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x1
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x0
	v_add_f32_e32 v57, v57, v21
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b3_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	.Lmx_b3_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b4:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b4_fold
	.Lmx_b4_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b4_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b4_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b4_s2
	s_branch .Lmx_b4_s3
	.Lmx_b4_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v24, v64 :: v_dual_add_f32 v65, v65, v25
	s_branch .Lmx_b4_group_done
	.Lmx_b4_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v24, v66 :: v_dual_add_f32 v67, v67, v25
	s_branch .Lmx_b4_group_done
	.Lmx_b4_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v24, v68 :: v_dual_add_f32 v69, v69, v25
	s_branch .Lmx_b4_group_done
	.Lmx_b4_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v24, v70 :: v_dual_add_f32 v71, v71, v25
	.Lmx_b4_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b4_group
	.Lmx_b4_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v68, v68, v70 :: v_dual_add_f32 v69, v69, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	ds_swizzle_b32 v16, v40 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v41 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v48 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v49 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v56 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v57 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x7
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x7
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x7
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v65, v65, v23
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x7
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v65, v65, v23
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	global_store_b32 v11, v64, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b4_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	global_store_b32 v11, v65, s[8:9] offset:4
	.Lmx_b4_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b5:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b5_fold
	.Lmx_b5_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b5_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b5_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b5_s2
	s_branch .Lmx_b5_s3
	.Lmx_b5_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v24, v64 :: v_dual_add_f32 v65, v65, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v24, v72 :: v_dual_add_f32 v73, v73, v25
	s_branch .Lmx_b5_group_done
	.Lmx_b5_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v24, v66 :: v_dual_add_f32 v67, v67, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v24, v74 :: v_dual_add_f32 v75, v75, v25
	s_branch .Lmx_b5_group_done
	.Lmx_b5_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v24, v68 :: v_dual_add_f32 v69, v69, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v24, v76 :: v_dual_add_f32 v77, v77, v25
	s_branch .Lmx_b5_group_done
	.Lmx_b5_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v24, v70 :: v_dual_add_f32 v71, v71, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v24, v78 :: v_dual_add_f32 v79, v79, v25
	.Lmx_b5_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b5_group
	.Lmx_b5_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v68, v68, v70 :: v_dual_add_f32 v69, v69, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v76, v76, v78 :: v_dual_add_f32 v77, v77, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v76 :: v_dual_add_f32 v73, v73, v77
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
	s_wait_dscnt 0x9
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x8
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x7
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x6
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x5
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x4
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x1
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x9
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x8
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x7
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x6
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x5
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x4
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x1
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x9
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x8
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x7
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x6
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x5
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x4
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x1
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x9
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x8
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x7
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x6
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x5
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x4
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x1
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v73, v73, v25
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0x9
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0x8
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x7
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x6
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x5
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x4
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x1
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v73, v73, v25
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	global_store_b32 v11, v64, s[8:9]
	global_store_b32 v12, v72, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b5_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	global_store_b32 v11, v65, s[8:9] offset:4
	global_store_b32 v12, v73, s[8:9] offset:4
	.Lmx_b5_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b6:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b6_fold
	.Lmx_b6_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b6_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b6_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b6_s2
	s_branch .Lmx_b6_s3
	.Lmx_b6_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v24, v64 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v24, v72 :: v_dual_add_f32 v73, v73, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v24, v80 :: v_dual_add_f32 v81, v81, v25
	s_branch .Lmx_b6_group_done
	.Lmx_b6_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v24, v66 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v24, v74 :: v_dual_add_f32 v75, v75, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v24, v82 :: v_dual_add_f32 v83, v83, v25
	s_branch .Lmx_b6_group_done
	.Lmx_b6_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v24, v68 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v24, v76 :: v_dual_add_f32 v77, v77, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v24, v84 :: v_dual_add_f32 v85, v85, v25
	s_branch .Lmx_b6_group_done
	.Lmx_b6_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v24, v70 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v24, v78 :: v_dual_add_f32 v79, v79, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v24, v86 :: v_dual_add_f32 v87, v87, v25
	.Lmx_b6_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b6_group
	.Lmx_b6_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v68, v68, v70 :: v_dual_add_f32 v69, v69, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v76, v76, v78 :: v_dual_add_f32 v77, v77, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v76 :: v_dual_add_f32 v73, v73, v77
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v84, v84, v86 :: v_dual_add_f32 v85, v85, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
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
	s_wait_dscnt 0xb
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xb
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xb
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xb
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v81, v81, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xb
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v81, v81, v27
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	global_store_b32 v11, v64, s[8:9]
	global_store_b32 v12, v72, s[8:9]
	global_store_b32 v13, v80, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b6_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	global_store_b32 v11, v65, s[8:9] offset:4
	global_store_b32 v12, v73, s[8:9] offset:4
	global_store_b32 v13, v81, s[8:9] offset:4
	.Lmx_b6_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b7:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b7_fold
	.Lmx_b7_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b7_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b7_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b7_s2
	s_branch .Lmx_b7_s3
	.Lmx_b7_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v24, v64 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v24, v72 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v24, v80 :: v_dual_add_f32 v81, v81, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v24, v88 :: v_dual_add_f32 v89, v89, v25
	s_branch .Lmx_b7_group_done
	.Lmx_b7_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v24, v66 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v24, v74 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v24, v82 :: v_dual_add_f32 v83, v83, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v24, v90 :: v_dual_add_f32 v91, v91, v25
	s_branch .Lmx_b7_group_done
	.Lmx_b7_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v24, v68 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v24, v76 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v24, v84 :: v_dual_add_f32 v85, v85, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v24, v92 :: v_dual_add_f32 v93, v93, v25
	s_branch .Lmx_b7_group_done
	.Lmx_b7_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v24, v70 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v24, v78 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v24, v86 :: v_dual_add_f32 v87, v87, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v24, v94 :: v_dual_add_f32 v95, v95, v25
	.Lmx_b7_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b7_group
	.Lmx_b7_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v68, v68, v70 :: v_dual_add_f32 v69, v69, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v76, v76, v78 :: v_dual_add_f32 v77, v77, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v76 :: v_dual_add_f32 v73, v73, v77
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v84, v84, v86 :: v_dual_add_f32 v85, v85, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v88, v88, v90 :: v_dual_add_f32 v89, v89, v91
	v_dual_add_f32 v92, v92, v94 :: v_dual_add_f32 v93, v93, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
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
	s_wait_dscnt 0xd
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xc
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xb
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xa
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x9
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x8
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x5
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x4
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x3
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x2
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x1
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xd
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xc
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xb
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xa
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x9
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x8
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x5
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x4
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x3
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x2
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x1
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xd
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xc
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xb
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xa
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x9
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x8
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x5
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x4
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x3
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x2
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x1
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xd
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xc
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xb
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xa
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x9
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x8
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x5
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x4
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x3
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x2
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x1
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v89, v89, v29
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xd
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xc
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xb
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xa
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0x9
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0x8
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x5
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x4
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x3
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x2
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x1
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v89, v89, v29
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	global_store_b32 v11, v64, s[8:9]
	global_store_b32 v12, v72, s[8:9]
	global_store_b32 v13, v80, s[8:9]
	global_store_b32 v14, v88, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b7_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	global_store_b32 v11, v65, s[8:9] offset:4
	global_store_b32 v12, v73, s[8:9] offset:4
	global_store_b32 v13, v81, s[8:9] offset:4
	global_store_b32 v14, v89, s[8:9] offset:4
	.Lmx_b7_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_b8:
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
	s_mov_b32 s24, 0
	s_cmp_eq_u32 s14, 0
	s_cbranch_scc1 .Lmx_b8_fold
	.Lmx_b8_group:
	s_mul_i32 s18, s24, 0x88
	s_lshl_b32 s19, s24, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s18, v1
	v_lshlrev_b32_e32 v2, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s18, v2
	buffer_load_b32 v8, v1, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v2, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v9, v1, s[20:23], s17 offen scope:SCOPE_DEV
	buffer_load_b32 v11, v2, s[20:23], s17 offen offset:8 scope:SCOPE_DEV
	v_lshlrev_b32_e32 v3, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v3, s19, v3
	global_load_b128 v[105:108], v3, s[28:29]
	global_load_b128 v[109:112], v3, s[28:29] offset:16
	global_load_b128 v[113:116], v3, s[30:31]
	global_load_b128 v[117:120], v3, s[30:31] offset:16
	s_wait_loadcnt 0x6
	v_bfe_u32 v16, v10, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v16, v8, v16, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x4
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
	s_and_b32 s25, s24, 3
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc1 .Lmx_b8_s0
	s_cmp_eq_u32 s25, 1
	s_cbranch_scc1 .Lmx_b8_s1
	s_cmp_eq_u32 s25, 2
	s_cbranch_scc1 .Lmx_b8_s2
	s_branch .Lmx_b8_s3
	.Lmx_b8_s0:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v24, v40 :: v_dual_add_f32 v41, v41, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v24, v48 :: v_dual_add_f32 v49, v49, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v24, v56 :: v_dual_add_f32 v57, v57, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v24, v64 :: v_dual_add_f32 v65, v65, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v24, v72 :: v_dual_add_f32 v73, v73, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v24, v80 :: v_dual_add_f32 v81, v81, v25
	global_load_b128 v[113:116], v3, s[42:43]
	global_load_b128 v[117:120], v3, s[42:43] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v24, v88 :: v_dual_add_f32 v89, v89, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v96, v24, v96 :: v_dual_add_f32 v97, v97, v25
	s_branch .Lmx_b8_group_done
	.Lmx_b8_s1:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v42, v24, v42 :: v_dual_add_f32 v43, v43, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v50, v24, v50 :: v_dual_add_f32 v51, v51, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v58, v24, v58 :: v_dual_add_f32 v59, v59, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v66, v24, v66 :: v_dual_add_f32 v67, v67, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v74, v24, v74 :: v_dual_add_f32 v75, v75, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v82, v24, v82 :: v_dual_add_f32 v83, v83, v25
	global_load_b128 v[113:116], v3, s[42:43]
	global_load_b128 v[117:120], v3, s[42:43] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v90, v24, v90 :: v_dual_add_f32 v91, v91, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v98, v24, v98 :: v_dual_add_f32 v99, v99, v25
	s_branch .Lmx_b8_group_done
	.Lmx_b8_s2:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v44, v24, v44 :: v_dual_add_f32 v45, v45, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v52, v24, v52 :: v_dual_add_f32 v53, v53, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v60, v24, v60 :: v_dual_add_f32 v61, v61, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v68, v24, v68 :: v_dual_add_f32 v69, v69, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v76, v24, v76 :: v_dual_add_f32 v77, v77, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v84, v24, v84 :: v_dual_add_f32 v85, v85, v25
	global_load_b128 v[113:116], v3, s[42:43]
	global_load_b128 v[117:120], v3, s[42:43] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v92, v24, v92 :: v_dual_add_f32 v93, v93, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v100, v24, v100 :: v_dual_add_f32 v101, v101, v25
	s_branch .Lmx_b8_group_done
	.Lmx_b8_s3:
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v46, v24, v46 :: v_dual_add_f32 v47, v47, v25
	global_load_b128 v[105:108], v3, s[32:33]
	global_load_b128 v[109:112], v3, s[32:33] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v54, v24, v54 :: v_dual_add_f32 v55, v55, v25
	global_load_b128 v[113:116], v3, s[34:35]
	global_load_b128 v[117:120], v3, s[34:35] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v62, v24, v62 :: v_dual_add_f32 v63, v63, v25
	global_load_b128 v[105:108], v3, s[36:37]
	global_load_b128 v[109:112], v3, s[36:37] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v70, v24, v70 :: v_dual_add_f32 v71, v71, v25
	global_load_b128 v[113:116], v3, s[38:39]
	global_load_b128 v[117:120], v3, s[38:39] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v78, v24, v78 :: v_dual_add_f32 v79, v79, v25
	global_load_b128 v[105:108], v3, s[40:41]
	global_load_b128 v[109:112], v3, s[40:41] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v86, v24, v86 :: v_dual_add_f32 v87, v87, v25
	global_load_b128 v[113:116], v3, s[42:43]
	global_load_b128 v[117:120], v3, s[42:43] offset:16
	s_wait_loadcnt 0x2
	v_dual_mul_f32 v24, v16, v105 :: v_dual_mul_f32 v25, v27, v106
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v106 :: v_dual_fmac_f32 v25, v26, v105
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v107 :: v_dual_fmac_f32 v25, v28, v107
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v108 :: v_dual_fmac_f32 v25, v29, v108
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v109 :: v_dual_fmac_f32 v25, v30, v109
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v110 :: v_dual_fmac_f32 v25, v31, v110
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v111 :: v_dual_fmac_f32 v25, v32, v111
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v112 :: v_dual_fmac_f32 v25, v33, v112
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v94, v24, v94 :: v_dual_add_f32 v95, v95, v25
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v24, v16, v113 :: v_dual_mul_f32 v25, v27, v114
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v17, v114 :: v_dual_fmac_f32 v25, v26, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v18, v115 :: v_dual_fmac_f32 v25, v28, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v19, v116 :: v_dual_fmac_f32 v25, v29, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v20, v117 :: v_dual_fmac_f32 v25, v30, v117
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v21, v118 :: v_dual_fmac_f32 v25, v31, v118
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v22, v119 :: v_dual_fmac_f32 v25, v32, v119
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v24, v23, v120 :: v_dual_fmac_f32 v25, v33, v120
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v102, v24, v102 :: v_dual_add_f32 v103, v103, v25
	.Lmx_b8_group_done:
	s_add_co_i32 s24, s24, 1
	s_cmp_lt_u32 s24, s14
	s_cbranch_scc1 .Lmx_b8_group
	.Lmx_b8_fold:
	v_dual_add_f32 v40, v40, v42 :: v_dual_add_f32 v41, v41, v43
	v_dual_add_f32 v44, v44, v46 :: v_dual_add_f32 v45, v45, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v40, v40, v44 :: v_dual_add_f32 v41, v41, v45
	v_dual_add_f32 v48, v48, v50 :: v_dual_add_f32 v49, v49, v51
	v_dual_add_f32 v52, v52, v54 :: v_dual_add_f32 v53, v53, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v48, v48, v52 :: v_dual_add_f32 v49, v49, v53
	v_dual_add_f32 v56, v56, v58 :: v_dual_add_f32 v57, v57, v59
	v_dual_add_f32 v60, v60, v62 :: v_dual_add_f32 v61, v61, v63
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v56, v56, v60 :: v_dual_add_f32 v57, v57, v61
	v_dual_add_f32 v64, v64, v66 :: v_dual_add_f32 v65, v65, v67
	v_dual_add_f32 v68, v68, v70 :: v_dual_add_f32 v69, v69, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v72, v72, v74 :: v_dual_add_f32 v73, v73, v75
	v_dual_add_f32 v76, v76, v78 :: v_dual_add_f32 v77, v77, v79
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v72, v72, v76 :: v_dual_add_f32 v73, v73, v77
	v_dual_add_f32 v80, v80, v82 :: v_dual_add_f32 v81, v81, v83
	v_dual_add_f32 v84, v84, v86 :: v_dual_add_f32 v85, v85, v87
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v88, v88, v90 :: v_dual_add_f32 v89, v89, v91
	v_dual_add_f32 v92, v92, v94 :: v_dual_add_f32 v93, v93, v95
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v96, v96, v98 :: v_dual_add_f32 v97, v97, v99
	v_dual_add_f32 v100, v100, v102 :: v_dual_add_f32 v101, v101, v103
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v96, v96, v100 :: v_dual_add_f32 v97, v97, v101
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
	s_wait_dscnt 0xf
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v96, v96, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xf
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v96, v96, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xf
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v96, v96, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xf
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v96, v96, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v97, v97, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
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
	s_wait_dscnt 0xf
	v_add_f32_e32 v40, v40, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v41, v41, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v48, v48, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v49, v49, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v56, v56, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v57, v57, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v64, v64, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v65, v65, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v72, v72, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v73, v73, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v80, v80, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v81, v81, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v96, v96, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v97, v97, v31
	v_cmpx_eq_u32_e32 0, v0
	s_mul_i32 s47, s45, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s47
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v12, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v13, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v14, s47
	s_mul_i32 s47, s45, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v15, s47
	global_store_b32 v8, v40, s[8:9]
	global_store_b32 v9, v48, s[8:9]
	global_store_b32 v10, v56, s[8:9]
	global_store_b32 v11, v64, s[8:9]
	global_store_b32 v12, v72, s[8:9]
	global_store_b32 v13, v80, s[8:9]
	global_store_b32 v14, v88, s[8:9]
	global_store_b32 v15, v96, s[8:9]
	s_cmp_eq_u32 s13, s12
	s_cbranch_scc1 .Lmx_b8_stored
	global_store_b32 v8, v41, s[8:9] offset:4
	global_store_b32 v9, v49, s[8:9] offset:4
	global_store_b32 v10, v57, s[8:9] offset:4
	global_store_b32 v11, v65, s[8:9] offset:4
	global_store_b32 v12, v73, s[8:9] offset:4
	global_store_b32 v13, v81, s[8:9] offset:4
	global_store_b32 v14, v89, s[8:9] offset:4
	global_store_b32 v15, v97, s[8:9] offset:4
	.Lmx_b8_stored:
	s_wait_storecnt 0x0
	s_branch .Lmx_end
	.Lmx_end:
	s_endpgm
.Lgemv_mq4g256v2_multirow_r2_xbatch_pm_end:
.size gemv_mq4g256v2_multirow_r2_xbatch_pm, .Lgemv_mq4g256v2_multirow_r2_xbatch_pm_end-gemv_mq4g256v2_multirow_r2_xbatch_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_multirow_r2_xbatch_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
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
	.amdhsa_system_sgpr_workgroup_id_y 0
	.amdhsa_system_sgpr_workgroup_id_z 0
	.amdhsa_system_sgpr_workgroup_info 0
	.amdhsa_system_vgpr_workitem_id 0
	.amdhsa_next_free_vgpr 121
	.amdhsa_next_free_sgpr 48
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_multirow_r2_xbatch_pm_end-gemv_mq4g256v2_multirow_r2_xbatch_pm)<<4)&4080)>>4
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
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y
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
        .name: B
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_multirow_r2_xbatch_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 50
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_multirow_r2_xbatch_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 121
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
