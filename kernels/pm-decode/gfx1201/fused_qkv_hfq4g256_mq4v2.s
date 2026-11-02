.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected fused_qkv_mq4g256v2
.globl fused_qkv_mq4g256v2
.p2align 8
.type fused_qkv_mq4g256v2,@function
fused_qkv_mq4g256v2:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b128 s[12:15], s[0:1], 0x20
	s_load_b128 s[16:19], s[0:1], 0x30
	s_load_b64 s[20:21], s[0:1], 0x40
	v_lshlrev_b32_e32 v1, 2, v0
	v_add_nc_u32_e32 v1, 8, v1
	v_lshlrev_b32_e32 v2, 5, v0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e64 v3, 4, 0, vcc_lo
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	s_wait_kmcnt 0x0
	s_add_co_i32 s22, s18, s19
	s_add_co_i32 s23, s22, s20
	s_cmp_ge_i32 ttmp9, s23
	s_cbranch_scc1 .Lqkv_done
	s_sub_co_i32 s30, ttmp9, s18
	s_cmp_lt_i32 ttmp9, s18
	s_cselect_b32 s24, ttmp9, s30
	s_cselect_b32 s26, s4, s6
	s_cselect_b32 s27, s5, s7
	s_cselect_b32 s28, s12, s14
	s_cselect_b32 s29, s13, s15
	s_sub_co_i32 s30, ttmp9, s22
	s_cmp_lt_i32 ttmp9, s22
	s_cselect_b32 s24, s24, s30
	s_cselect_b32 s26, s26, s8
	s_cselect_b32 s27, s27, s9
	s_cselect_b32 s28, s28, s16
	s_cselect_b32 s29, s29, s17
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s31, s21, 8
	s_lshr_b32 s34, s31, 2
	s_and_b32 s35, s31, 3
	s_mul_i32 s32, s31, 0x88
	s_ashr_i32 s33, s32, 31
	s_mul_u64 s[32:33], s[32:33], s[24:25]
	s_add_nc_u64 s[32:33], s[26:27], s[32:33]
	s_mov_b64 s[36:37], s[32:33]
	s_and_b32 s37, s37, 0xffff
	s_mov_b32 s38, -1
	s_mov_b32 s39, 0x31004000
	s_mov_b32 s40, 0
	s_cmp_eq_u32 s34, 0
	s_cbranch_scc1 .Lqkv_tails
	.Lqkv_quads:
	s_clause 0x9
	buffer_load_b64 v[8:9], v31, s[36:39], s40 offen scope:SCOPE_DEV
	buffer_load_b32 v10, v31, s[36:39], s40 offen offset:136 scope:SCOPE_DEV
	buffer_load_b32 v11, v31, s[36:39], s40 offen offset:140 scope:SCOPE_DEV
	buffer_load_b32 v12, v31, s[36:39], s40 offen offset:272 scope:SCOPE_DEV
	buffer_load_b32 v13, v31, s[36:39], s40 offen offset:276 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v31, s[36:39], s40 offen offset:408 scope:SCOPE_DEV
	buffer_load_b32 v20, v1, s[36:39], s40 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v1, s[36:39], s40 offen offset:136 scope:SCOPE_DEV
	buffer_load_b32 v22, v1, s[36:39], s40 offen offset:272 scope:SCOPE_DEV
	buffer_load_b32 v23, v1, s[36:39], s40 offen offset:408 scope:SCOPE_DEV
	s_clause 0x7
	global_load_b128 v[32:35], v2, s[10:11]
	global_load_b128 v[40:43], v2, s[10:11] offset:1024
	global_load_b128 v[48:51], v2, s[10:11] offset:2048
	global_load_b128 v[56:59], v2, s[10:11] offset:3072
	global_load_b128 v[36:39], v2, s[10:11] offset:16
	global_load_b128 v[44:47], v2, s[10:11] offset:1040
	global_load_b128 v[52:55], v2, s[10:11] offset:2064
	global_load_b128 v[60:63], v2, s[10:11] offset:3088
	s_add_co_i32 s40, s40, 0x220
	s_sub_co_i32 s34, s34, 1
	s_cmp_lg_u32 s34, 0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e64 v4, v9, v8, vcc_lo
	s_wait_loadcnt 0xc
	v_cndmask_b32_e64 v5, v11, v10, vcc_lo
	v_cndmask_b32_e64 v6, v13, v12, vcc_lo
	v_cndmask_b32_e64 v7, v15, v14, vcc_lo
	s_wait_loadcnt 0x8
	v_bfe_u32 v65, v22, 4, 4
	v_and_b32_e32 v64, 15, v22
	v_bfe_u32 v66, v22, 8, 4
	v_bfe_u32 v67, v22, 12, 4
	v_bfe_u32 v68, v22, 16, 4
	v_bfe_u32 v70, v22, 24, 4
	v_lshrrev_b32_e32 v71, 28, v22
	v_bfe_u32 v69, v22, 20, 4
	v_bfe_u32 v73, v23, 4, 4
	v_and_b32_e32 v72, 15, v23
	v_bfe_u32 v74, v23, 8, 4
	v_bfe_u32 v75, v23, 12, 4
	v_bfe_u32 v76, v23, 16, 4
	v_bfe_u32 v78, v23, 24, 4
	v_lshrrev_b32_e32 v79, 28, v23
	v_bfe_u32 v77, v23, 20, 4
	v_bfe_u32 v9, v20, 4, 4
	v_and_b32_e32 v8, 15, v20
	v_bfe_u32 v10, v20, 8, 4
	v_bfe_u32 v11, v20, 12, 4
	v_bfe_u32 v12, v20, 16, 4
	v_bfe_u32 v14, v20, 24, 4
	v_lshrrev_b32_e32 v15, 28, v20
	v_bfe_u32 v13, v20, 20, 4
	v_bfe_u32 v17, v21, 4, 4
	v_and_b32_e32 v16, 15, v21
	v_bfe_u32 v18, v21, 8, 4
	v_bfe_u32 v19, v21, 12, 4
	v_bfe_u32 v20, v21, 16, 4
	v_bfe_u32 v22, v21, 24, 4
	v_lshrrev_b32_e32 v23, 28, v21
	v_bfe_u32 v21, v21, 20, 4
	v_cvt_f32_ubyte0_e32 v8, v8
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v64, v64
	v_cvt_f32_ubyte0_e32 v72, v72
	v_cvt_f32_ubyte0_e32 v9, v9
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v65, v65
	v_cvt_f32_ubyte0_e32 v73, v73
	v_cvt_f32_ubyte0_e32 v10, v10
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v66, v66
	v_cvt_f32_ubyte0_e32 v74, v74
	v_cvt_f32_ubyte0_e32 v11, v11
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v67, v67
	v_cvt_f32_ubyte0_e32 v75, v75
	v_cvt_f32_ubyte0_e32 v12, v12
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v68, v68
	v_cvt_f32_ubyte0_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v13, v13
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v69, v69
	v_cvt_f32_ubyte0_e32 v77, v77
	v_cvt_f32_ubyte0_e32 v14, v14
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v70, v70
	v_cvt_f32_ubyte0_e32 v78, v78
	v_cvt_f32_ubyte0_e32 v15, v15
	v_cvt_f32_ubyte0_e32 v23, v23
	v_cvt_f32_ubyte0_e32 v71, v71
	v_cvt_f32_ubyte0_e32 v79, v79
	v_fma_mix_f32 v8, v4, v8, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v6, v64, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v7, v72, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v9, v4, v9, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v6, v65, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v7, v73, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v10, v4, v10, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v6, v66, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v7, v74, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v11, v4, v11, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v6, v67, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v7, v75, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v12, v4, v12, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v68, v6, v68, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v7, v76, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v13, v4, v13, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v69, v6, v69, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v7, v77, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v14, v4, v14, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v6, v70, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v78, v7, v78, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v15, v4, v15, v4 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v6, v71, v6 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v79, v7, v79, v7 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v4, v9, v33
	v_mul_f32_e32 v6, v65, v49
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v8, v32 :: v_dual_mul_f32 v5, v17, v41
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v64, v48 :: v_dual_mul_f32 v7, v73, v57
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v10, v34 :: v_dual_fmac_f32 v5, v16, v40
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v66, v50 :: v_dual_fmac_f32 v7, v72, v56
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v11, v35 :: v_dual_fmac_f32 v5, v18, v42
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v67, v51 :: v_dual_fmac_f32 v7, v74, v58
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v12, v36 :: v_dual_fmac_f32 v5, v19, v43
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v68, v52 :: v_dual_fmac_f32 v7, v75, v59
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v13, v37 :: v_dual_fmac_f32 v5, v20, v44
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v69, v53 :: v_dual_fmac_f32 v7, v76, v60
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v14, v38 :: v_dual_fmac_f32 v5, v21, v45
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v70, v54 :: v_dual_fmac_f32 v7, v77, v61
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v4, v15, v39 :: v_dual_fmac_f32 v5, v22, v46
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v6, v71, v55 :: v_dual_fmac_f32 v7, v78, v62
	v_fmac_f32_e32 v5, v23, v47
	v_fmac_f32_e32 v7, v79, v63
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v26, v26, v4 :: v_dual_add_f32 v27, v27, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v28, v28, v6 :: v_dual_add_f32 v29, v29, v7
	v_add_nc_u32_e32 v2, 0x1000, v2
	s_cbranch_scc1 .Lqkv_quads
	.Lqkv_tails:
	s_cmp_lt_u32 s35, 1
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33]
	global_load_b32 v6, v7, s[32:33]
	global_load_b128 v[8:11], v2, s[10:11]
	global_load_b128 v[12:15], v2, s[10:11] offset:16
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v26, v26, v24
	s_cmp_lt_u32 s35, 2
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33] offset:136
	global_load_b32 v6, v7, s[32:33] offset:136
	global_load_b128 v[8:11], v2, s[10:11] offset:1024
	global_load_b128 v[12:15], v2, s[10:11] offset:1040
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v27, v27, v24
	s_cmp_lt_u32 s35, 3
	s_cbranch_scc1 .Lqkv_reduce
	v_add_nc_u32_e32 v4, s40, v3
	v_add_nc_u32_e32 v7, s40, v1
	global_load_b32 v5, v4, s[32:33] offset:272
	global_load_b32 v6, v7, s[32:33] offset:272
	global_load_b128 v[8:11], v2, s[10:11] offset:2048
	global_load_b128 v[12:15], v2, s[10:11] offset:2064
	s_wait_loadcnt 0x2
	v_bfe_u32 v16, v6, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_fma_mix_f32 v16, v5, v16, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v17, v6, 4, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_fma_mix_f32 v17, v5, v17, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v18, v6, 8, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	v_fma_mix_f32 v18, v5, v18, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v19, v6, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v19, v5, v19, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v20, v6, 16, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_fma_mix_f32 v20, v5, v20, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v21, v6, 20, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_fma_mix_f32 v21, v5, v21, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v22, v6, 24, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_fma_mix_f32 v22, v5, v22, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v23, v6, 28, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v23, v5, v23, v5 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v24, v17, v9
	v_fmac_f32_e32 v24, v16, v8
	v_fmac_f32_e32 v24, v18, v10
	v_fmac_f32_e32 v24, v19, v11
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v24, v20, v12
	v_fmac_f32_e32 v24, v21, v13
	v_fmac_f32_e32 v24, v22, v14
	v_fmac_f32_e32 v24, v23, v15
	v_add_f32_e32 v28, v28, v24
	.Lqkv_reduce:
	v_add_f32_e32 v24, v26, v27
	v_add_f32_e32 v25, v28, v29
	v_add_f32_e32 v24, v24, v25
	ds_swizzle_b32 v30, v24 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v31, 0, 8, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v31, 0, 4, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v31, 0, 2, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v31, 0, 1, vcc_lo
	v_add_lshl_u32 v31, v31, v0, 2
	ds_bpermute_b32 v30, v31, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v30
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .Lqkv_done
	s_lshl_b64 s[24:25], s[24:25], 2
	s_add_nc_u64 s[32:33], s[28:29], s[24:25]
	v_mov_b32_e32 v1, 0
	global_store_b32 v1, v24, s[32:33]
	s_wait_storecnt 0x0
	.Lqkv_done:
	s_endpgm
.Lfused_qkv_mq4g256v2_end:
.size fused_qkv_mq4g256v2, .Lfused_qkv_mq4g256v2_end-fused_qkv_mq4g256v2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel fused_qkv_mq4g256v2
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 72
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
	.amdhsa_next_free_vgpr 80
	.amdhsa_next_free_sgpr 42
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lfused_qkv_mq4g256v2_end-fused_qkv_mq4g256v2)<<4)&4080)>>4
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
        .name: A_q
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A_k
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: A_v
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_q
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_k
        .offset: 40
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: y_v
        .offset: 48
        .size: 8
        .value_kind: global_buffer
      - 
        .name: q_m
        .offset: 56
        .size: 4
        .value_kind: by_value
      - 
        .name: k_m
        .offset: 60
        .size: 4
        .value_kind: by_value
      - 
        .name: v_m
        .offset: 64
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 68
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .max_flat_workgroup_size: 32
    .name: fused_qkv_mq4g256v2
    .private_segment_fixed_size: 0
    .sgpr_count: 44
    .sgpr_spill_count: 0
    .symbol: fused_qkv_mq4g256v2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 80
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
