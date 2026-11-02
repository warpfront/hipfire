.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_multirow_r2
.globl gemv_mq4g256v2_multirow_r2
.p2align 8
.type gemv_mq4g256v2_multirow_r2,@function
gemv_mq4g256v2_multirow_r2:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_lshl_b32 s3, ttmp9, 1
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s3, s10
	s_cbranch_scc1 .Lmultirow_exit
	s_lshr_b32 s12, s11, 8
	s_mul_i32 s13, s12, 0x88
	s_add_co_i32 s14, s3, 1
	s_cmp_lt_i32 s14, s10
	s_cselect_b32 s14, s14, s3
	s_mul_i32 s16, s3, s13
	s_mov_b32 s17, 0
	s_add_nc_u64 s[16:17], s[4:5], s[16:17]
	s_mul_i32 s18, s14, s13
	s_mov_b32 s19, 0
	s_add_nc_u64 s[18:19], s[4:5], s[18:19]
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	s_mov_b32 s15, 0
	s_and_b32 s23, s12, -4
	s_cmp_eq_u32 s23, 0
	s_cbranch_scc1 .Lmultirow_tail
	.Lmultirow_quad:
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v20, v7, v20
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v24, v24, v8
	s_add_co_i32 s15, s15, 1
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v21, v7, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v25, v25, v8
	s_add_co_i32 s15, s15, 1
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v22, v7, v22
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v26, v26, v8
	s_add_co_i32 s15, s15, 1
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v23, v7, v23
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v27, v27, v8
	s_add_co_i32 s15, s15, 1
	s_cmp_lt_u32 s15, s23
	s_cbranch_scc1 .Lmultirow_quad
	.Lmultirow_tail:
	s_cmp_lt_u32 s15, s12
	s_cbranch_scc0 .Lmultirow_reduce
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v20, v7, v20
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v24, v24, v8
	s_add_co_i32 s15, s15, 1
	s_cmp_lt_u32 s15, s12
	s_cbranch_scc0 .Lmultirow_reduce
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v21, v7, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v25, v25, v8
	s_add_co_i32 s15, s15, 1
	s_cmp_lt_u32 s15, s12
	s_cbranch_scc0 .Lmultirow_reduce
	s_mul_i32 s22, s15, 0x88
	s_lshl_b32 s21, s15, 10
	v_lshrrev_b32_e32 v1, 4, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v1, 2, v1
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v4, v1, s[16:17]
	global_load_b32 v5, v1, s[18:19]
	v_lshlrev_b32_e32 v1, 2, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s22, v1
	global_load_b32 v2, v1, s[16:17] offset:8
	global_load_b32 v3, v1, s[18:19] offset:8
	v_lshlrev_b32_e32 v1, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v1, s21, v1
	global_load_b128 v[12:15], v1, s[6:7]
	global_load_b128 v[16:19], v1, s[6:7] offset:16
	s_wait_loadcnt 0x3
	v_bfe_u32 v6, v2, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, v12, v6
	v_bfe_u32 v6, v2, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v13, v6
	v_bfe_u32 v6, v2, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v14, v6
	v_bfe_u32 v6, v2, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v15, v6
	v_bfe_u32 v6, v2, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v16, v6
	v_bfe_u32 v6, v2, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v17, v6
	v_bfe_u32 v6, v2, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v18, v6
	v_bfe_u32 v6, v2, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v4, v6, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, v19, v6
	v_bfe_u32 v6, v3, 4, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, v13, v6
	v_bfe_u32 v6, v3, 0, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v12, v6
	v_bfe_u32 v6, v3, 8, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v14, v6
	v_bfe_u32 v6, v3, 12, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v15, v6
	v_bfe_u32 v6, v3, 16, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v16, v6
	v_bfe_u32 v6, v3, 20, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v17, v6
	v_bfe_u32 v6, v3, 24, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v18, v6
	v_bfe_u32 v6, v3, 28, 4
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f32_ubyte0_e32 v6, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v6, v5, v6, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, v19, v6
	v_add_f32_e32 v22, v7, v22
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v26, v26, v8
	s_add_co_i32 s15, s15, 1
	.Lmultirow_reduce:
	v_add_f32_e32 v28, v20, v21
	v_add_f32_e32 v30, v22, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	ds_swizzle_b32 v30, v28 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v28, v28, v30
	v_add_f32_e32 v29, v24, v25
	v_add_f32_e32 v31, v26, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v31
	ds_swizzle_b32 v31, v29 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v29, v29, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v9, v9, v0, 2
	ds_bpermute_b32 v30, v9, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v31, v9, v29
	s_wait_dscnt 0x0
	v_add_f32_e32 v29, v29, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v9, v9, v0, 2
	ds_bpermute_b32 v30, v9, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v31, v9, v29
	s_wait_dscnt 0x0
	v_add_f32_e32 v29, v29, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v9, v9, v0, 2
	ds_bpermute_b32 v30, v9, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v31, v9, v29
	s_wait_dscnt 0x0
	v_add_f32_e32 v29, v29, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v9, v9, v0, 2
	ds_bpermute_b32 v30, v9, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v30
	ds_bpermute_b32 v31, v9, v29
	s_wait_dscnt 0x0
	v_add_f32_e32 v29, v29, v31
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .Lmultirow_exit
	s_lshl_b32 s20, s3, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v1, s20
	global_store_b32 v1, v28, s[8:9]
	s_cmp_eq_u32 s14, s3
	s_cbranch_scc1 .Lmultirow_exit
	global_store_b32 v1, v29, s[8:9] offset:4
	s_wait_storecnt 0x0
	.Lmultirow_exit:
	s_endpgm
.Lgemv_mq4g256v2_multirow_r2_end:
.size gemv_mq4g256v2_multirow_r2, .Lgemv_mq4g256v2_multirow_r2_end-gemv_mq4g256v2_multirow_r2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_multirow_r2
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 32
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
	.amdhsa_next_free_vgpr 32
	.amdhsa_next_free_sgpr 24
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_multirow_r2_end-gemv_mq4g256v2_multirow_r2)<<4)&4080)>>4
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_multirow_r2
    .private_segment_fixed_size: 0
    .sgpr_count: 26
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_multirow_r2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 32
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
