.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected fused_gate_up_mq4g256v2
.globl fused_gate_up_mq4g256v2
.p2align 8
.type fused_gate_up_mq4g256v2,@function
fused_gate_up_mq4g256v2:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b128 s[16:19], s[0:1], 0x20
	s_load_b32 s28, s[0:1], 0x30
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 ttmp9, s18
	s_cselect_b32 s20, 0, s18
	s_cselect_b32 s4, s4, s6
	s_cselect_b32 s5, s5, s7
	s_cselect_b32 s10, s10, s16
	s_cselect_b32 s11, s11, s17
	s_sub_co_i32 s30, ttmp9, s20
	s_ashr_i32 s31, s30, 31
	s_ashr_i32 s21, s28, 8
	s_mul_i32 s22, s21, 0x88
	s_ashr_i32 s23, s22, 31
	s_mul_u64 s[22:23], s[22:23], s[30:31]
	s_add_nc_u64 s[12:13], s[4:5], s[22:23]
	s_and_b32 s13, s13, 0xffff
	s_mov_b32 s14, -1
	s_mov_b32 s15, 0x31004000
	v_lshlrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v2, 5, v0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	s_mov_b32 s20, 0
	s_mov_b32 s22, 0
	s_mov_b32 s23, 0
	s_lshr_b32 s28, s21, 2
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc1 .Lgate_up_tail
	.Lgate_up_quads:
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v3, s22
	v_add_nc_u32_e32 v28, s22, v1
	v_add_nc_u32_e32 v29, s23, v2
	buffer_load_b32 v8, v3, s[12:15], null offen scope:SCOPE_DEV
	buffer_load_b32 v9, v3, s[12:15], null offen offset:4 scope:SCOPE_DEV
	buffer_load_b32 v16, v28, s[12:15], null offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v10, v3, s[12:15], null offen offset:136 scope:SCOPE_DEV
	buffer_load_b32 v11, v3, s[12:15], null offen offset:140 scope:SCOPE_DEV
	buffer_load_b32 v17, v28, s[12:15], null offen offset:144 scope:SCOPE_DEV
	buffer_load_b64 v[12:13], v3, s[12:15], null offen offset:272 scope:SCOPE_DEV
	buffer_load_b32 v18, v28, s[12:15], null offen offset:280 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v3, s[12:15], null offen offset:408 scope:SCOPE_DEV
	buffer_load_b32 v19, v28, s[12:15], null offen offset:416 scope:SCOPE_DEV
	global_load_b128 v[32:35], v29, s[8:9]
	global_load_b128 v[36:39], v29, s[8:9] offset:16
	global_load_b128 v[40:43], v29, s[8:9] offset:1024
	global_load_b128 v[44:47], v29, s[8:9] offset:1040
	global_load_b128 v[48:51], v29, s[8:9] offset:2048
	global_load_b128 v[52:55], v29, s[8:9] offset:2064
	global_load_b128 v[56:59], v29, s[8:9] offset:3072
	global_load_b128 v[60:63], v29, s[8:9] offset:3088
	v_cmp_lt_u32_e64 s24, v0, 16
	s_wait_loadcnt 0x10
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v8, v9, v8, s24
	s_wait_loadcnt 0x8
	v_cndmask_b32_e64 v10, v11, v10, s24
	v_cndmask_b32_e64 v12, v13, v12, s24
	v_cndmask_b32_e64 v14, v15, v14, s24
	v_bfe_u32 v4, v16, 4, 4
	v_bfe_u32 v5, v17, 4, 4
	v_bfe_u32 v6, v18, 4, 4
	v_bfe_u32 v7, v19, 4, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_mul_f32_e32 v20, v33, v4
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v21, v41, v5
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v22, v49, v6
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v23, v57, v7
	v_bfe_u32 v4, v16, 0, 4
	v_bfe_u32 v5, v17, 0, 4
	v_bfe_u32 v6, v18, 0, 4
	v_bfe_u32 v7, v19, 0, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v32, v4
	v_fmac_f32_e32 v21, v40, v5
	v_fmac_f32_e32 v22, v48, v6
	v_fmac_f32_e32 v23, v56, v7
	v_bfe_u32 v4, v16, 8, 4
	v_bfe_u32 v5, v17, 8, 4
	v_bfe_u32 v6, v18, 8, 4
	v_bfe_u32 v7, v19, 8, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v34, v4
	v_fmac_f32_e32 v21, v42, v5
	v_fmac_f32_e32 v22, v50, v6
	v_fmac_f32_e32 v23, v58, v7
	v_bfe_u32 v4, v16, 12, 4
	v_bfe_u32 v5, v17, 12, 4
	v_bfe_u32 v6, v18, 12, 4
	v_bfe_u32 v7, v19, 12, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v35, v4
	v_fmac_f32_e32 v21, v43, v5
	v_fmac_f32_e32 v22, v51, v6
	v_fmac_f32_e32 v23, v59, v7
	v_bfe_u32 v4, v16, 16, 4
	v_bfe_u32 v5, v17, 16, 4
	v_bfe_u32 v6, v18, 16, 4
	v_bfe_u32 v7, v19, 16, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v36, v4
	v_fmac_f32_e32 v21, v44, v5
	v_fmac_f32_e32 v22, v52, v6
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v23, v60, v7
	v_bfe_u32 v4, v16, 20, 4
	v_bfe_u32 v5, v17, 20, 4
	v_bfe_u32 v6, v18, 20, 4
	v_bfe_u32 v7, v19, 20, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v37, v4
	v_fmac_f32_e32 v21, v45, v5
	v_fmac_f32_e32 v22, v53, v6
	v_fmac_f32_e32 v23, v61, v7
	v_bfe_u32 v4, v16, 24, 4
	v_bfe_u32 v5, v17, 24, 4
	v_bfe_u32 v6, v18, 24, 4
	v_bfe_u32 v7, v19, 24, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v38, v4
	v_fmac_f32_e32 v21, v46, v5
	v_fmac_f32_e32 v22, v54, v6
	v_fmac_f32_e32 v23, v62, v7
	v_bfe_u32 v4, v16, 28, 4
	v_bfe_u32 v5, v17, 28, 4
	v_bfe_u32 v6, v18, 28, 4
	v_bfe_u32 v7, v19, 28, 4
	v_cvt_f32_ubyte0_e32 v4, v4
	v_cvt_f32_ubyte0_e32 v5, v5
	v_cvt_f32_ubyte0_e32 v6, v6
	v_cvt_f32_ubyte0_e32 v7, v7
	v_fma_mix_f32 v4, v8, v4, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v5, v10, v5, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v6, v12, v6, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v7, v14, v7, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v20, v39, v4
	v_fmac_f32_e32 v21, v47, v5
	v_fmac_f32_e32 v22, v55, v6
	v_fmac_f32_e32 v23, v63, v7
	v_add_f32_e32 v24, v24, v20
	v_add_f32_e32 v25, v25, v21
	v_add_f32_e32 v26, v26, v22
	v_add_f32_e32 v27, v27, v23
	s_add_co_i32 s20, s20, 4
	s_add_co_i32 s22, s22, 0x220
	s_add_co_i32 s23, s23, 0x1000
	s_add_co_i32 s28, s28, -1
	s_cmp_eq_u32 s28, 0
	s_cbranch_scc0 .Lgate_up_quads
	.Lgate_up_tail:
	s_cmp_lt_u32 s20, s21
	s_cbranch_scc0 .Lgate_up_fold
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v3, s22
	v_add_nc_u32_e32 v28, s22, v1
	v_add_nc_u32_e32 v29, s23, v2
	buffer_load_b64 v[6:7], v3, s[12:15], null offen scope:SCOPE_DEV
	v_cmp_lt_u32_e64 s24, v0, 16
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, v7, v6, s24
	buffer_load_b32 v4, v28, s[12:15], null offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[8:11], v29, s[8:9]
	global_load_b128 v[12:15], v29, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v4, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v4, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v4, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v4, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v4, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v4, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v4, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v4, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v6, v9, v17
	v_fmac_f32_e32 v6, v8, v16
	v_fmac_f32_e32 v6, v10, v18
	v_fmac_f32_e32 v6, v11, v19
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v6, v12, v20
	v_fmac_f32_e32 v6, v13, v21
	v_fmac_f32_e32 v6, v14, v22
	v_fmac_f32_e32 v6, v15, v23
	v_add_f32_e32 v24, v24, v6
	s_add_co_i32 s20, s20, 1
	s_add_co_i32 s22, s22, 0x88
	s_add_co_i32 s23, s23, 0x400
	s_cmp_lt_u32 s20, s21
	s_cbranch_scc0 .Lgate_up_fold
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v3, s22
	v_add_nc_u32_e32 v28, s22, v1
	v_add_nc_u32_e32 v29, s23, v2
	buffer_load_b64 v[6:7], v3, s[12:15], null offen scope:SCOPE_DEV
	v_cmp_lt_u32_e64 s24, v0, 16
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, v7, v6, s24
	buffer_load_b32 v4, v28, s[12:15], null offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[8:11], v29, s[8:9]
	global_load_b128 v[12:15], v29, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v4, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v4, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v4, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v4, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v4, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v4, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v4, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v4, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v6, v9, v17
	v_fmac_f32_e32 v6, v8, v16
	v_fmac_f32_e32 v6, v10, v18
	v_fmac_f32_e32 v6, v11, v19
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v6, v12, v20
	v_fmac_f32_e32 v6, v13, v21
	v_fmac_f32_e32 v6, v14, v22
	v_fmac_f32_e32 v6, v15, v23
	v_add_f32_e32 v25, v25, v6
	s_add_co_i32 s20, s20, 1
	s_add_co_i32 s22, s22, 0x88
	s_add_co_i32 s23, s23, 0x400
	s_cmp_lt_u32 s20, s21
	s_cbranch_scc0 .Lgate_up_fold
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v3, s22
	v_add_nc_u32_e32 v28, s22, v1
	v_add_nc_u32_e32 v29, s23, v2
	buffer_load_b64 v[6:7], v3, s[12:15], null offen scope:SCOPE_DEV
	v_cmp_lt_u32_e64 s24, v0, 16
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v5, v7, v6, s24
	buffer_load_b32 v4, v28, s[12:15], null offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[8:11], v29, s[8:9]
	global_load_b128 v[12:15], v29, s[8:9] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v4, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v4, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v4, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v4, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v4, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v4, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v4, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v4, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v6, v9, v17
	v_fmac_f32_e32 v6, v8, v16
	v_fmac_f32_e32 v6, v10, v18
	v_fmac_f32_e32 v6, v11, v19
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v6, v12, v20
	v_fmac_f32_e32 v6, v13, v21
	v_fmac_f32_e32 v6, v14, v22
	v_fmac_f32_e32 v6, v15, v23
	v_add_f32_e32 v26, v26, v6
	s_add_co_i32 s20, s20, 1
	s_add_co_i32 s22, s22, 0x88
	s_add_co_i32 s23, s23, 0x400
	.Lgate_up_fold:
	v_add_f32_e32 v24, v24, v25
	v_add_f32_e32 v26, v27, v26
	v_add_f32_e32 v24, v24, v26
	ds_swizzle_b32 v6, v24 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 24
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 8, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 28
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 4, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 30
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 2, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmp_lt_u32_e64 s24, v0, 31
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, 0, 1, s24
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v6, v3, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v6
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .Lgate_up_end
	s_lshl_b64 s[30:31], s[30:31], 2
	s_add_nc_u64 s[10:11], s[10:11], s[30:31]
	v_mov_b32_e32 v3, 0
	global_store_b32 v3, v24, s[10:11]
	s_wait_storecnt 0x0
	.Lgate_up_end:
	s_endpgm
.Lfused_gate_up_mq4g256v2_end:
.size fused_gate_up_mq4g256v2, .Lfused_gate_up_mq4g256v2_end-fused_gate_up_mq4g256v2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel fused_gate_up_mq4g256v2
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 52
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
	.amdhsa_next_free_vgpr 64
	.amdhsa_next_free_sgpr 32
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lfused_gate_up_mq4g256v2_end-fused_gate_up_mq4g256v2)<<4)&4080)>>4
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
        .name: A_gate
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A_up
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_gate
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_up
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - 
        .name: gate_m
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: up_m
        .offset: 44
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 48
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .max_flat_workgroup_size: 32
    .name: fused_gate_up_mq4g256v2
    .private_segment_fixed_size: 0
    .sgpr_count: 32
    .sgpr_spill_count: 0
    .symbol: fused_gate_up_mq4g256v2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 64
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
